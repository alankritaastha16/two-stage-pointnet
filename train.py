'''
Example of training a semantic segmentation model (uses the SemanticKITTI
dataset).
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--datadir', default='/data')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--feature-transform', action='store_true')
parser.add_argument('--cache', action='store_true')
parser.add_argument('--small', action='store_true')
parser.add_argument('--big-model')
parser.add_argument('--region-strategy')
args = parser.parse_args()

from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import numpy as np
import pnets as pn
import myaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

# load the dataset with augmentation

if args.big_model:
    big_model = torch.load(args.big_model)

if args.small:
    if args.region_strategy == 'random-clip':
        region_clip = myaug.RandomClip(0.1)
    if args.region_strategy == 'hardest-region':
        region_clip = myaug.HardestRegionModel(0.1, big_model, 0.5)
    aug = pn.aug.Compose(
        region_clip,
        pn.aug.Resample(args.npoints),
        pn.aug.Normalize(),
        pn.aug.Jitter(),
        pn.aug.RandomRotation('Z', 0, 2*np.pi),
    )
else:
    aug = pn.aug.Compose(
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
#tr = torch.utils.data.Subset(tr, range(10))  # DEBUG
num_workers = 0 if args.region_strategy == 'hardest-region' else 4
tr = DataLoader(tr, 32, True, num_workers=num_workers, pin_memory=True)

# create the model
model = pn.pointnet.PointNetSeg(K).to(device)
summary(model)
print('model output:', model(torch.ones((10, 3, 2500), device=device))[0].shape)

opt = torch.optim.Adam(model.parameters(), 1e-3)
ce_loss = torch.nn.CrossEntropyLoss()

# train the model
model.train()
for epoch in range(args.epochs):
    KK = []
    KK_pred = []
    print(f'* Epoch {epoch+1} / {args.epochs}')
    tic = time()
    avg_loss = 0
    avg_acc = 0
    for P, Y in tqdm(tr):
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
        KK.append(Y.view(-1))
        KK_pred.append(K_pred.detach().view(-1))
    toc = time()
    print(f'- {toc-tic:.1f}s - Loss: {avg_loss} - Acc: {avg_acc}')
    KK = torch.cat(KK)
    KK_pred = torch.cat(KK_pred)
    #print('IoU:', pn.metrics.IoU(KK_pred, KK, K).cpu().numpy())
    print('mIoU:', pn.metrics.mIoU(KK_pred, KK, K).cpu().numpy())

if args.small:
    fname = f'model-{args.dataset}-small-{args.region_strategy}.pth'
else:
    fname = f'model-{args.dataset}.pth'
torch.save(model, fname)
