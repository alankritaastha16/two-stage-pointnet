import argparse
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import pnets as pn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import myaug
from averageprob import confs_average, preds_to_confs

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()
parser.add_argument('model', help='model path')
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--datadir', default='/data')
parser.add_argument('--cache', action='store_true')
#parser.add_argument('--clip-percentage', default=0.1, type=float)
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

aug = pn.aug.Compose(
    pn.aug.Resample(args.npoints),
    pn.aug.Normalize(),
    # data augmentation
    #pn.aug.Jitter(),
    #pn.aug.RandomRotation('Z', 0, 2*np.pi),
)
ts = getattr(pn.data, args.dataset)
ts = ts(args.datadir, 'val', None if args.cache else aug)
K = ts.nclasses
if args.cache:
    ts = pn.data.Cache(ts, aug)
#ts = torch.utils.data.Subset(ts, range(10))  # DEBUG
ts = DataLoader(ts, 1, True, num_workers=4, pin_memory=True)

model = torch.load(args.model, map_location=device)
model.eval()
print(ts)
for points, segs in ts:
    #print(points)
    
    points_cuda = points.to(device)
    with torch.no_grad():
        segs_pred, _, _ = model(points_cuda)
    confs = preds_to_confs(segs_pred)
    print("confs:", confs.shape)
    confs_avg, points_per_region,xyz_start,xyz_end= confs_average(points_cuda, confs, 0.25)
    #print("confs_avg:", confs_avg.shape)
    #print("points_per_region:", points_per_region.keys())
    confs_avg = confs_avg.cpu().numpy()
    ind = np.unravel_index(np.nanargmin(confs_avg, axis=None), confs_avg.shape)
    print("ind:", ind)
    start=[xyz_start[ind][0].cpu().numpy(),xyz_start[ind][1].cpu().numpy(),xyz_start[ind][2].cpu().numpy()]
    print("start:", start)
    end=[xyz_end[ind][0].cpu().numpy(),xyz_end[ind][1].cpu().numpy(),xyz_end[ind][2].cpu().numpy()]
    print("end:", end)
    # color is the probability associated to the maximum probability
    msk=points_per_region[ind][0].cpu().numpy()
    print("msk:", msk.shape, msk.sum())
    critical_points = points[:, : , msk]
    #S = S[msk]
    segs_pred = F.softmax(segs_pred, 1)
    corners=[(start[0],start[1],start[2]),(start[0],start[1],end[2]),(start[0],end[1],end[2]),(start[0],end[1],start[2]),(end[0],end[1],start[2]),(end[0],start[1],start[2]),(end[0],start[1],end[2]),(end[0],end[1],end[2])]
    print("corners:",corners)
    verts = [[corners[0],corners[1],corners[2],corners[3]], [corners[0],corners[1],corners[6],corners[5]], [corners[1], corners[2],corners[7], corners[6]], [corners[0], corners[3], corners[4], corners[5]], [corners[4], corners[5], corners[6], corners[7]], [corners[3], corners[2], corners[7], corners[4]]]
    print("verts:",verts)
    fig = plt.figure()
    
    point = points[0]
    critical_point = critical_points[0]
    seg = segs[0]
    seg_pred = segs_pred[0]

    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(point[0], point[1], point[2], c="k", marker='.', alpha=0.2, edgecolors='none')
    
    ax.scatter(critical_point[0], critical_point[1], critical_point[2], c="r", marker='.', alpha=1,       edgecolors='none')

   
    ax.add_collection3d(Poly3DCollection(verts, edgecolors='black',alpha=.1))
    
    """
    ax.plot((mins[0], maxs[0]), (mins[1], mins[1]), (mins[2], mins[2]), c="r")
    ax.plot((mins[0], maxs[0]), (maxs[1], maxs[1]), (maxs[2], maxs[2]), c="r")

    ax.plot((mins[0], mins[0]), (mins[1], maxs[1]), (mins[2], mins[2]), c="r")
    ax.plot((maxs[0], maxs[0]), (mins[1], maxs[1]), (maxs[2], maxs[2]), c="r")

    ax.plot((mins[0], mins[0]), (mins[1], mins[1]), (mins[2], maxs[2]), c="r")
    ax.plot((maxs[0], maxs[0]), (maxs[1], maxs[1]), (mins[2], maxs[2]), c="r")
    """
    
    ax.set_title("black")

    # image color=ground-truth segmentation
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(point[0], point[1], point[2], c=seg, marker='.', alpha=0.2, edgecolors='none')
    
    ax.set_title("ground-truth")

    # image color=predicted segmentation
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(point[0], point[1], point[2], c=seg_pred.argmax(0).cpu(), marker='.', alpha=0.2, edgecolors='none')
    ax.set_title("predicted classes")
    ax.add_collection3d(Poly3DCollection(verts, edgecolors='black',alpha=.1))

    print('classes:', np.bincount(seg_pred.argmax(0).cpu()))

    # image color=probabilities
    # color is the probability associated to the true class
    color = seg_pred.amax(0).cpu()
    ax = fig.add_subplot(224, projection='3d')
    p = ax.scatter(point[0], point[1], point[2], c=color, marker='.', alpha=0.2, edgecolors='none')
    ax.add_collection3d(Poly3DCollection(verts,alpha=.1))
    ax.set_title("confidence")
    fig.colorbar(p)
    #print()
    plt.show()

"""
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
xxxxxx


point_np=point
point=torch.from_numpy(point)
point = point.transpose(1, 0).contiguous()
point = point.view(1, point.shape[0], point.shape[1])
point=point.to(device)
print(point.shape)
save_path = args.model
model = pn.pointnet.PointNetSeg(K).to(device)
model = torch.load(save_path)
model.eval()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
#gt = cmap[seg - 1, :]

pred, _, _ = model(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
pn.plot.plot3d(point_np, pred_color)
pn.plot.show()
"""
