import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--datadir', default='/data')
parser.add_argument('--npoints', default=2500, type=int)
args = parser.parse_args()

from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import numpy as np
import pnets as pn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

# load the dataset with augmentation
aug = pn.aug.Compose(
    pn.aug.Resample(args.npoints),
    pn.aug.Normalize(),
)
ts = getattr(pn.data, args.dataset)
ts = ts(args.datadir, 'train', aug)
K = ts.nclasses
ts = DataLoader(ts, 1, True, num_workers=4, pin_memory=True)

# load the model
model = torch.load(args.model).to(device)

# evaluate the model
model.eval()

KK = []
KK_pred = []
tic = time()
avg_acc = 0
for P, Y in tqdm(ts):
    P = P.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        Y_pred, _, _ = model(P)

    P_pred = F.softmax(Y_pred, 1)
    K_pred = P_pred.argmax(1)
    avg_acc += float((Y == K_pred).float().mean()) / len(ts)
    KK.append(Y.view(-1))
    KK_pred.append(K_pred.detach().view(-1))

toc = time()
print(f'Time: {toc-tic:.1f}s')
print(f'Acc: {avg_acc}')
KK = torch.cat(KK)
KK_pred = torch.cat(KK_pred)
#print('IoU:', pn.metrics.IoU(KK_pred, KK, K).cpu().numpy())
print('mIoU:', pn.metrics.mIoU(KK_pred, KK, K).cpu().numpy())
