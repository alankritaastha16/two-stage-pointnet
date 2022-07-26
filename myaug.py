'''
Point-cloud augmentation pipeline routines.

Each augmentation routine `f(P, S)` must have as input the points and segmentation, and output the same. The segmentation can be `None` if there are no segmentations.
'''

import numpy as np
import torch
from averageprob import confs_average, preds_to_confs
import pnets as pn

def HardestRegionModel(clip_percentage, big_model, threshold):
    def f(P, S):
        device = next(big_model.parameters()).device
        big_model.eval()
        P_tensor = torch.tensor(P[None, :, :], device=device)
        with torch.no_grad():
            preds, _, _ = big_model(P_tensor)
        confs = preds_to_confs(preds)
        confs_avg, points_per_region = confs_average(P_tensor, confs, clip_percentage)
        confs_avg = confs_avg.cpu().numpy()
        #confs_avg[isnan(confs_avg)] = np.inf
        ind = np.unravel_index(np.nanargmin(confs_avg, axis=None), confs_avg.shape)
        print(ind,confs_avg[ind])
        msk=points_per_region[ind][0].cpu().numpy()
        print(msk)
        P = P[:, msk]
        S = S[msk]
        #pn.plot.plot3d(P)
        #pn.plot.show()
        return P, S
    return f

def RandomClip(clip_percentage):
    def f(P, S):
        Xstep=(max(P[0])-min(P[0]))/clip_percentage        
        Ystep=(max(P[1])-min(P[1]))/clip_percentage
        Zstep=(max(P[2])-min(P[2]))/clip_percentage
        X, Y, Z = P[:,np.random.choice(P.shape[1])]
        msk = np.logical_and(np.logical_and((np.logical_and
                    (P[0,:]>=X-Xstep/2,P[0,:]<X+Xstep/2)),
                    (np.logical_and(P[1,:]>=Y-Ystep/2, P[1,:]<Y+Ystep/2))),
                    np.logical_and(P[2,:]>=Z-Zstep/2, P[2,:]<Z+Zstep/2))
        P = P[:, msk]
        S = S[msk]
        return P, S
    return f

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    args = parser.parse_args()
    import pnets as pn
    t = pn.aug.Compose(
        pn.aug.RandomRotation('Z', 0, 2*np.pi),
        RandomClip(20),
    )
    ds = pn.data.ICCV17ShapeNet(args.datadir, 'train', t,True)
    P, Y = ds[1]
    #print(ds.labels[Y])
    pn.plot.plot3d(P)
    pn.plot.show()
