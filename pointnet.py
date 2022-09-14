'''
A PointNet implementation for classification and segmentation.
We performed some changes on the implementations, namely no activation function
is applied on the outputs (i.e. logits are produced).
Original author: https://github.com/fxia22/pointnet.pytorch
PointNet paper: https://arxiv.org/abs/1612.00593
'''

import torch
from torch import nn
import torch.nn.functional as F

def feature_transform_regularizer(M):
    ''' You may use this regularizer in your loss, as recommended by the PointNet paper. The feature transform is returned by the PointNet model. '''
    I = torch.eye(M.shape[1], device=M.device)[None]
    loss = torch.mean(torch.norm(torch.bmm(M, M.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class BatchNormIfSample(nn.BatchNorm1d):
    ''' Layer that applies batch-norm only if batchsize > 1, in order to avoid errors otherwise. '''
    def forward(self, x):
        if x.shape[0] == 1:
            return x
        return super().forward(x)

class STNkd(nn.Module):
    '''PointNet basic module.'''
    def __init__(self, k):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = BatchNormIfSample(64)
        self.bn2 = BatchNormIfSample(128)
        self.bn3 = BatchNormIfSample(1024)
        self.bn4 = BatchNormIfSample(512)
        self.bn5 = BatchNormIfSample(256)

        self.k = k

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batchsize,1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    '''PointNet backbone.'''
    def __init__(self, size=6, global_feat=True, feature_transform=False):
        super().__init__()
        self.size = size
        self.stn = STNkd(3)
        self.conv1 = torch.nn.Conv1d(3, 2**size, 1)
        self.conv2 = torch.nn.Conv1d(2**size, 2**(size+1), 1)
        self.conv3 = torch.nn.Conv1d(2**(size+1), 2**(size+4), 1)
        self.bn1 = BatchNormIfSample(2**size)
        self.bn2 = BatchNormIfSample(2**(size+1))
        self.bn3 = BatchNormIfSample(2**(size+4))
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(2**size)

    def forward(self, x):
        n_pts = x.shape[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 2**(self.size+4))
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 2**(self.size+4), 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    '''Classification PointNet.'''
    def __init__(self, k, feature_transform=False):
        super().__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = BatchNormIfSample(512)
        self.bn2 = BatchNormIfSample(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat

class PointNetSeg(nn.Module):
    '''Segmentation PointNet.'''
    def __init__(self, k, size=6, feature_transform=False):
        super().__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(size, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(2**(size+4)+2**size, 2**(size+3), 1)
        self.conv2 = torch.nn.Conv1d(2**(size+3), 2**(size+2), 1)
        self.conv3 = torch.nn.Conv1d(2**(size+2), 2**(size+1), 1)
        self.conv4 = torch.nn.Conv1d(2**(size+1), self.k, 1)
        self.bn1 = BatchNormIfSample(2**(size+3))
        self.bn2 = BatchNormIfSample(2**(size+2))
        self.bn3 = BatchNormIfSample(2**(size+1))

    def forward(self, x):
        batchsize, _, n_pts = x.shape
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x, trans, trans_feat

if __name__ == '__main__':
    from torch.autograd import Variable
    sim_data = Variable(torch.rand(32, 3, 2500))
    cls = PointNetCls(5)
    out, _, _ = cls(sim_data)
    print('class', out.shape)
