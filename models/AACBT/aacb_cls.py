import torch.nn as nn
import torch.nn.functional as F
from models.AACBT.pointnet2_utils import InputEmbedding, PointNetSetAbstractionMsg_xyz, PointNetSetAbstractionMsg_others, PointNetSetAbstraction, Global_Transformer
from models.AACBT.CrossAttention import CrossAttention
import torch
import numpy as np

class get_model(nn.Module):
    def __init__(self,num_class, geometric_channel, feature_channel, sampling_rate, npoints):
        super(get_model, self).__init__()

        in_channel_others = len(feature_channel)
        in_channel_xyz = len(geometric_channel)

        self.channel_idx = geometric_channel+feature_channel
        self.geometric_channel = geometric_channel
        self.feature_channel = feature_channel
        embeded_channel = 64

        sampling_points = []

        for i in range(2): # sampling for 2 times
            s_r = int(npoints/np.power(sampling_rate, i+1))
            sampling_points.append(s_r)

        self.embedding_xyz = InputEmbedding([4], [16], in_channel_xyz,[[32, 32, embeded_channel]])
        self.embedding_others = InputEmbedding([4], [16], in_channel_others,[[32, 32, embeded_channel]])
        
        self.crossattention1 = CrossAttention(embeded_channel, num_heads=2)

        self.sa1_xyz = PointNetSetAbstractionMsg_xyz(sampling_points[0], [25,225,625], [32, 64, 128], embeded_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.global_sa1_xyz = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=sampling_points[0], in_channel=320, out_channels=320, layers=1, num_heads=8, head_dim=40)
        self.sa1_others = PointNetSetAbstractionMsg_others(sampling_points[0], [25,225,625], [32, 64, 128], embeded_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.global_sa1_others = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=sampling_points[0], in_channel=320, out_channels=320, layers=1, num_heads=8, head_dim=40)

        self.crossattention2 = CrossAttention(320, num_heads = 8)

        self.sa2_xyz = PointNetSetAbstractionMsg_xyz(sampling_points[1], [50, 450, 1250], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.global_sa2_xyz = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=sampling_points[1], in_channel=640, out_channels=640, layers=1, num_heads=16, head_dim=40)
        self.sa2_others = PointNetSetAbstractionMsg_others(sampling_points[1], [50, 450, 1250], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.global_sa2_others = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=sampling_points[1], in_channel=640, out_channels=640, layers=1, num_heads=16, head_dim=40)
        
        self.crossattention3 = CrossAttention(640, num_heads = 16)
        
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.fc1 = nn.Linear(1984, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape

        if (0 not in self.channel_idx) or (1 not in self.channel_idx) or (2 not in self.channel_idx):
            raise Exception("No engough coordinate features!!! Please try again")

        others = xyz[:, self.feature_channel, :]
        geometric = xyz[:, self.geometric_channel, :]
        xyz = xyz[:, :3, :]
        # print(xyz.shape, geometric.shape)
        xyz, embeded_geometric = self.embedding_xyz(xyz, geometric)
        _, embeded_others = self.embedding_others(xyz, others)

        crossfeatures1 = self.crossattention1(embeded_others,embeded_geometric)

        l1_xyz, l1_xyzpoints, l1_grouped_xyzpoints_list, s_index_1 = self.sa1_xyz(xyz, crossfeatures1)
        _, g1_xyzpoints = self.global_sa1_xyz(l1_xyz, l1_xyzpoints, l1_grouped_xyzpoints_list) # b d n
        _, l1_others, l1_grouped_otherspoints_list= self.sa1_others(xyz, embeded_others, s_index_1)
        _, g1_others = self.global_sa1_others(l1_xyz, l1_others, l1_grouped_otherspoints_list) # b d n

        crossfeatures2 = self.crossattention2(g1_others, g1_xyzpoints)
        global_feats1 = self.alpha*torch.max(crossfeatures2, 2)[0] # b d

        l2_xyz, l2_xyzpoints, l2_grouped_points_list, s_index_2 = self.sa2_xyz(l1_xyz, crossfeatures2)
        _, g2_xyzpoints = self.global_sa2_xyz(l2_xyz, l2_xyzpoints, l2_grouped_points_list) # b d n
        _, l2_others, l2_grouped_otherspoints_list= self.sa2_others(l1_xyz, g1_others, s_index_2)
        _, g2_others = self.global_sa2_others(l2_xyz, l2_others, l2_grouped_otherspoints_list) # b d n

        crossfeatures3 = self.crossattention3(g2_others, g2_xyzpoints)
        global_feats2 = self.beta*torch.max(crossfeatures3, 2)[0]# b d

        l3_xyz, l3_points = self.sa3(l2_xyz, crossfeatures3)

        global_feats3 = l3_points.view(B, 1024)

        x = torch.cat((global_feats1, global_feats2, global_feats3), dim=-1) # 320+640+1024 
        x = self.drop1(F.relu(self.bn1(self.fc1(x)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, gold, smoothing=True):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss


class get_loss_weighted(nn.Module):
    def __init__(self):
      super(get_loss_weighted, self).__init__()
    def forward(self, pred, target, weight):
        total_loss = F.cross_entropy(pred, target, weight)
        #total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss