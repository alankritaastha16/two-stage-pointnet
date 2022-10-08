'''
Example of training a semantic segmentation model (uses the SemanticKITTI
dataset).
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('split', choices=['first', 'second'])
parser.add_argument('--datadir', default='/data')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--feature-transform', action='store_true')
parser.add_argument('--cache', action='store_true')
parser.add_argument('--small', action='store_true')
parser.add_argument('--big-model')
parser.add_argument('--size', default=6, type=int)
parser.add_argument('--region-strategy')
parser.add_argument('--conf',default=0.5,type=float)
args = parser.parse_args()
    
from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import numpy as np
import pnets as pn
import pointnet
import myaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

# load the dataset with augmentation

def MaxPoints(n):
    def f(P, S):
        if len(S) > n:
            ix = np.random.choice(len(S), n, False)
            P = P[:, ix]
            S = S[ix]
        return P, S
    return f

if args.big_model:
    big_model = torch.load(args.big_model)

if args.small:
    if args.region_strategy == 'random-clip':
        region_clip = myaug.RandomClip(0.1)
    if args.region_strategy == 'hardest-region':
        region_clip = myaug.HardestRegion(0.1, big_model)
    if args.region_strategy == 'lessconf-points':
        region_clip = myaug.pointclouds_lessconf(big_model, args.conf)
    aug = pn.aug.Compose(
        #MaxPoints(10000),
        pn.aug.Normalize(),
        pn.aug.Resample(args.npoints),
        pn.aug.Jitter(),
        pn.aug.RandomRotation('Z', 0, 2*np.pi),
        region_clip,
    )
else:
    aug = pn.aug.Compose(
       # MaxPoints(10000),
        pn.aug.Resample(args.npoints),
        pn.aug.Normalize(),
        pn.aug.Jitter(),
        pn.aug.RandomRotation('Z', 0, 2*np.pi),
    )
tr = getattr(pn.data, args.dataset)
tr = tr(args.datadir, 'train', None if args.cache else aug)
K = tr.nclasses
if args.cache:
    tr = pn.data.Cache(tr, aug)

rand = np.random.RandomState(123)
ix = rand.choice(len(tr), len(tr), False)
if args.split == 'first':
	ix = ix[:len(ix)//2]
else:
	ix = ix[len(ix)//2:]
tr = torch.utils.data.Subset(tr, ix)

#tr = torch.utils.data.Subset(tr, range(10))  # DEBUG
num_workers = 0 if args.region_strategy == 'hardest-region' or 'lessconf-points' else 4
tr = DataLoader(tr, 1, True, num_workers=num_workers, pin_memory=True)

# create the model
model = pointnet.PointNetSeg(K, args.size).to(device)
summary(model)
#print('model output:', model(torch.ones((10, 3, 2500), device=device))[0].shape)

opt = torch.optim.Adam(model.parameters(), 1e-3)
ce_loss = torch.nn.CrossEntropyLoss()

# train the model
model.train()
for epoch in range(args.epochs):
    kk = torch.arange(K, device=device)[:, None]
    #KK = []
    #KK_pred = []
    cmii = torch.tensor([0]*K, device=device)
    cmij = torch.tensor([0]*K, device=device)
    cmki = torch.tensor([0]*K, device=device)
    print(f'* Epoch {epoch+1} / {args.epochs}')
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for P, Y in tqdm(tr):
        if P.shape[2] == 0:
            #print('No points - skip')
            continue
        P = P.to(device)
        Y = Y.to(device)

        Y_pred, trans, trans_feat = model(P)
        loss = ce_loss(Y_pred, Y)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        P_pred = F.softmax(Y_pred, 1)
        K_pred = P_pred.argmax(1)
        '''for p_pred, k_pred in zip(P_pred, K_pred):
            colors = p_pred[k_pred]
            # plot this
            #print(colors)
            #print(k_pred)'''
        avg_acc += float((Y == K_pred).float().mean()) / len(tr)
        #KK.append(Y.view(-1).cpu())
        #KK_pred.append(K_pred.detach().view(-1).cpu())
        y = Y.view(-1)
        ypred = K_pred.detach().view(-1)
        cmii += torch.logical_and(ypred == kk, y == kk).sum(1)
        cmij += torch.logical_and(ypred == kk, y != kk).sum(1)
        cmki += torch.logical_and(ypred != kk, y == kk).sum(1)

    toc = time()
    print(f'- {toc-tic:.1f}s - Loss: {avg_loss} - Acc: {avg_acc}')
    #KK = torch.cat(KK)
    #KK_pred = torch.cat(KK_pred)
    #print('IoU:', pn.metrics.IoU(KK_pred, KK, K).numpy())
    mIoU = (cmii / (cmii + cmij + cmki)).mean()
    print('mIoU:', mIoU.cpu())

if args.small:
    fname = f'model-{args.size}-{args.dataset}-small-{args.region_strategy}-{args.conf}.pth'
else:
    fname = f'model-{args.size}-{args.dataset}.pth'
torch.save(model.cpu(), fname)
