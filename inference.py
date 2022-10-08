'''
Example of validation of a dataset a shapenet model.
'''


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--datadir', default='/data')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--feature-transform', action='store_true')
parser.add_argument('--cache', action='store_true')
parser.add_argument('--big-model')
parser.add_argument('--small-model')
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
import myaug
import tqdm
from averageprob import preds_to_confs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

# load the dataset with augmentation
aug = pn.aug.Compose(
    pn.aug.Resample(args.npoints),
    pn.aug.Normalize(),
)
ts = getattr(pn.data, args.dataset)
ts = ts(args.datadir, 'val', aug)
K = ts.nclasses
ts = DataLoader(ts, 1, True, num_workers=4, pin_memory=True)

# load big_model:
big_model = torch.load(args.big_model, map_location=device)
big_model.eval()

#predict using big model
tic = time()
KK = []
KK_pred = []
avg_acc1 = 0
for P,S in ts:
    P = P.to(device)
    with torch.no_grad():
        preds, _, _ = big_model(P)
    
confs = preds_to_confs(preds).cpu().numpy()
        #print("confs:",confs.shape,confs)
ind = np.argwhere(confs < args.conf)
        #print("ind",ind.shape,ind,ind[0])
Points_lessconf = P[..., ind[:, 1]]
# load small_model:
small_model = torch.load(args.small_model,map_location=device)
small_model.eval()
Points_lessconf = Points_lessconf.to(device)
with torch.no_grad():
    Y_pred, _, _ = small_model(Points_lessconf)
preds[..., ind[:,1]]   =  Y_pred
toc = time()
print('time:', toc-tic)
P_pred = F.softmax(preds, 1)
K_pred = P_pred.argmax(1)
avg_acc2 = float((Y == K_pred).float().mean()) / len(tr)
KK = torch.cat(KK)
KK_pred = torch.cat(KK_pred)
#print('IoU:', pn.metrics.IoU(KK_pred, KK, K).cpu().numpy())
print('mIoU:', pn.metrics.mIoU(KK_pred, KK, K).cpu().numpy())





