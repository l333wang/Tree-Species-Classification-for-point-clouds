import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossAttention(nn.Module):
    def __init__(self, in_channel, num_heads = 4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.dim = in_channel
        self.out_dim = in_channel
        head_dim = in_channel // num_heads
        self.scale = head_dim ** -0.5

        self.wq_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wk_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wv_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wq_point = nn.Linear(in_channel, in_channel, bias=qkv_bias)
        self.wk_point = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wv_point = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_r2p = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.proj_p2r = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_emb, geo_emb): #[b, c, n] 
        rgb_emb = rgb_emb.permute(0,2,1)
        geo_emb = geo_emb.permute(0,2,1)#[b, N, C] 
        _, N_rgb, _ = rgb_emb.shape #2048
        B, N_geo, C = geo_emb.shape #2048
        # print("rgb, geo shape:{0},{1}".format(rgb_emb.shape,geo_emb.shape))
        #from point to rgb
        q_rgb = self.wq_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,8,2048,32
        k_point = self.wk_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,8,2048,32
        v_point = self.wv_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,8,2048,32
        # print("q k v shape:{0},{1},{2}".format(q.shape,k.shape,v.shape))
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # scale_p2r = np.sqrt(k_point.size(1))
        scale_p2r = self.scale
        
        attn_p2r = (q_rgb.transpose(-2, -1) @ k_point) * scale_p2r # B,8,32,32
        attn_p2r = self.softmax(attn_p2r) # B,4,32,32
        attn_p2r = self.attn_drop(attn_p2r)
        # print("attn shape: {0}".format(attn.shape))

        res_emb_p2r = (v_point @ attn_p2r).transpose(1,2).reshape(B, N_rgb, C) # B,2048,32*8
        
        res_emb_p2r = self.proj_drop(self.proj_p2r(res_emb_p2r)) # B, c, N_rgb
      
        #from rgb to point
        q_point = self.wq_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,N,C
        k_rgb = self.wk_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,C,N
        v_rgb = self.wv_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,N,C
        scale_r2p = self.scale

        attn_r2p = (q_point.transpose(-2, -1) @ k_rgb) * scale_r2p # B,N,N
        attn_r2p = self.softmax(attn_r2p) # B,N,N
        attn_r2p = self.attn_drop(attn_r2p)
        res_emb_r2p = (v_rgb @ attn_r2p).transpose(1,2).reshape(B, N_geo, C) # B, N C
        
        res_emb_r2p = self.proj_drop(self.proj_r2p(res_emb_r2p)) # B, N_geo, c

        rgb_emb_att = (rgb_emb + res_emb_p2r).permute(0, 2, 1)
        geo_emb_att = (geo_emb + res_emb_r2p).permute(0, 2, 1)
        
        # res = torch.cat([rgb_emb_att, geo_emb_att],dim=1) # B, C N
        res = rgb_emb_att+geo_emb_att # B, C N
        return res