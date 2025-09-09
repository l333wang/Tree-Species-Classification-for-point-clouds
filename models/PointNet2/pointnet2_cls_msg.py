import torch.nn as nn
import torch.nn.functional as F
from models.PointNet2.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch
import numpy as np

class get_model(nn.Module):
    def __init__(self,num_class, geometric_channel, feature_channel, sampling_rate, npoints):
        super(get_model, self).__init__()

        in_channel_others = len(feature_channel)
        in_channel_xyz = len(geometric_channel)
        in_channel = in_channel_others + in_channel_xyz - 3

        self.channel_idx = geometric_channel+feature_channel

        self.sa1 = PointNetSetAbstractionMsg(128, [0.1, 0.2, 0.4], [8, 16, 64], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(64, [0.2, 0.4, 0.8], [16, 32, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
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

        
        xyz = xyz[:, self.channel_idx, :]

        norm = xyz[:, 3:, :]
        xyz = xyz[:, :3, :]


        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss

# original loss
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