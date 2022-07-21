'''
Point-cloud augmentation pipeline routines.

Each augmentation routine `f(P, S)` must have as input the points and segmentation, and output the same. The segmentation can be `None` if there are no segmentations.
'''

import numpy as np

def RandomClip(clip_percentage):
    def f(P, S):
        '''
        P_=[]
        S_=[]
        Xmin= min(P[0])
        Xmax=max(P[0])
        Ymin= min(P[1])
        Ymax=max(P[1])
        Zmin= min(P[2])
        Zmax=max(P[2])
        '''
        Xstep=(max(P[0])-min(P[0]))/clip_percentage        
        Ystep=(max(P[1])-min(P[1]))/clip_percentage
        Zstep=(max(P[2])-min(P[2]))/clip_percentage
        X,Y,Z=P[:,np.random.choice(P.shape[1])]
        msk = np.logical_and(np.logical_and((np.logical_and
                    (P[0,:]>=X-Xstep/2,P[0,:]<X+Xstep/2)),
                    (np.logical_and(P[1,:]>=Y-Ystep/2, P[1,:]<Y+Ystep/2))),
                    np.logical_and(P[2,:]>=Z-Zstep/2, P[2,:]<Z+Zstep/2))
        P = P[:, msk]
        print(S.shape, S)
        if len(S)!=0:
            S = S[msk]
        return P, S
        """
        for i in range(len(msk)):
            if msk[i]:
                S_.append(S[i])
                P_.append(P[:,i])
        return P_,S_
        """
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
