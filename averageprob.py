import numpy as np 
import torch

def preds_to_confs(preds):
    confs = torch.softmax(preds, 1)  # (batchsize, nclasses, npoints)
    return confs.amax(1)

def confs_average(points, confs, clip_percentage=0.1):
    assert (1 / clip_percentage) % 10 == 0, clip_percentage
    assert len(points.shape) == 3, points.shape
    assert points.shape[1] == 3, points.shape
    assert len(confs.shape) == 2, confs.shape
    Xmin= torch.min(points[:,0,:])
    Xmax= torch.max(points[:,0,:])
    Ymin= torch.min(points[:,1,:])
    Ymax= torch.max(points[:,1,:])
    Zmin= torch.min(points[:,2,:])
    Zmax= torch.max(points[:,2,:])
    rect=torch.zeros(points.shape[0], int(1/clip_percentage), int(1/clip_percentage), int(1/clip_percentage))
    points_per_region = {}
    Xstep = (Xmax-Xmin) * clip_percentage         
    Ystep = (Ymax-Ymin) * clip_percentage
    Zstep = (Zmax-Zmin) * clip_percentage
    #msk= torch.logical_and(points[:,0,:]>=Xmin,points[:,0,:]<Xmin+Xstep)              
    #PTS_temp=points[msk]
    n = 0
    for i, x in enumerate(torch.arange(Xmin, Xmax-1e-6, Xstep)):
        for j, y in enumerate(torch.arange(Ymin, Ymax-1e-6, Ystep)):
            for k, z in enumerate(torch.arange(Zmin, Zmax-1e-6, Zstep)):
                msk = torch.logical_and(torch.logical_and((torch.logical_and
                    (points[:,0,:]>=x,points[:,0,:]<x+Xstep)),
                    (torch.logical_and(points[:,1,:]>=y, points[:,1,:]<y+Ystep))),
                    torch.logical_and(points[:,2,:]>=z, points[:,2,:]<z+Zstep))
                for batchi in range(points.shape[0]):
                    rect[batchi, i, j, k] = confs[batchi][msk[batchi]].mean()
                    points_per_region[(batchi, i, j, k)] = msk
    return rect, points_per_region

def draw_confidence(conf_rect):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = np.empty((*conf_rect.shape, 4), dtype=object)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if not torch.isnan(conf_rect[i, j, k]):
                    colors[i, j, k] = [1, 0, 0, conf_rect[i, j, k].item()]

    ax.voxels(torch.logical_not(torch.isnan(conf_rect)), facecolors=colors, edgecolor='k')
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if not torch.isnan(conf_rect[i, j, k]):
                    ax.text(i, j, k, int(conf_rect[i, j, k].item()*100), color='blue')
    plt.show()

if __name__ == '__main__':
    points=torch.tensor([[[ 0.3852, -0.3697, -0.3874, -0.7774, -0.6606,  0.4811],
             [-0.2323,  0.2297,  0.1892,   0.8413,  0.5990, -0.3459],
             [-0.6144, -0.5762, -0.9511,   0.0995,  0.3537,  0.2542]],

            [[ 0.8040,  0.5572,  0.7826,   -0.5616,  0.5698, -0.7944],
             [ 0.2257,  0.0533, -0.8678,   0.2448,  0.0533,  0.2257],
             [-0.7307, -0.0418,  1.0000,   -0.6209,  0.1810, -0.7646]],

            [[-0.6788,  0.1884, -0.2078,   -0.0579,  0.2399,  0.1884],
             [ 0.2657, -0.6609, -0.8714,   0.3361, -0.9297, -0.5301],
             [ 0.3740, -0.4624, -0.3765,   0.6591, -0.4414,  0.2631]],
             
            [[ 0.0221,  0.5545, -0.4202,   -0.6270,  0.8956, -0.0124],
             [-0.6085,  0.1792,  0.1379,   0.1792,  0.1379, -0.7394],
             [ 0.0374, -0.5157, -0.5823,  0.6876, -0.3541,  0.7749]],

            [[-0.0562,  0.2277,  0.4408,  0.5049,  0.0814, -0.2168],
             [ 0.1709,  0.2019,  0.1918,    0.1709,  0.1709,  0.1923],
             [ 0.7442,  0.9301,  0.1487,   -0.4607,  0.2573,  0.1404]],

            [[-0.5258,  0.0884,  0.2185,   0.2473,  0.6346,  0.4415],
             [ 0.2179, -0.0201, -0.0416,   -0.0945,  0.1565, -0.1995],
             [ 0.0037,  0.4335, -0.2142,   0.2215, -0.0647, -0.0539]]])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using', device)
    points = points.to(device)
    confs = torch.rand(6, 6, device=device)
     
    rect = confs_average(points, confs)
    draw_confidence(rect[0])
