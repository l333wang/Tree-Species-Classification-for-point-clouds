import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.PointNet.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, num_class, geometric_channel, feature_channel, sampling_rate, npoints):
        super(get_model, self).__init__()

        in_channel_others = len(feature_channel)
        in_channel_xyz = len(geometric_channel)
        channel = in_channel_others + in_channel_xyz

        self.channel_idx = geometric_channel+feature_channel

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):

        B, _, _ = x.shape

        if (0 not in self.channel_idx) or (1 not in self.channel_idx) or (2 not in self.channel_idx):
            raise Exception("No engough coordinate features!!! Please try again")

        
        x = x[:, self.channel_idx, :]

        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

# class get_loss(torch.nn.Module):
#     def __init__(self, mat_diff_loss_scale=0.001):
#         super(get_loss, self).__init__()
#         self.mat_diff_loss_scale = mat_diff_loss_scale

#     def forward(self, pred, target, trans_feat):
#         loss = F.nll_loss(pred, target)
#         mat_diff_loss = feature_transform_reguliarzer(trans_feat)

#         total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
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