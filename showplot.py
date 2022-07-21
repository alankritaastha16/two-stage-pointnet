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

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()
parser.add_argument('model', help='model path')
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--datadir', default='/data')
parser.add_argument('--cache', action='store_true')
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
ts = DataLoader(ts, 32, True, num_workers=4, pin_memory=True)

model = torch.load(args.model, map_location=device)
model.eval()
print(ts)
for points, segs in ts:
    print(points)
    points_cuda = points.to(device)
    with torch.no_grad():
        segs_pred, _, _ = model(points_cuda)

    segs_pred = F.softmax(segs_pred, 1)

    # color is the probability associated to the maximum probability
    #color = seg_pred[0].max(0)
    #print("color:", color.shape)

    for point, seg, seg_pred in zip(points, segs, segs_pred):
        fig = plt.figure()
        # image color=black
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(point[0], point[1], point[2], c="k", marker='.', alpha=0.2, edgecolors='none')
        ax.set_title("black")

        # image color=ground-truth segmentation
        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(point[0], point[1], point[2], c=seg, marker='.', alpha=0.2, edgecolors='none')
        ax.set_title("ground-truth")

        # image color=predicted segmentation
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(point[0], point[1], point[2], c=seg_pred.argmax(0).cpu(), marker='.', alpha=0.2, edgecolors='none')
        ax.set_title("predicted classes")

        print('classes:', np.bincount(seg_pred.argmax(0).cpu()))

        # image color=probabilities
        # color is the probability associated to the true class
        color = seg_pred.amax(0).cpu()
        ax = fig.add_subplot(224, projection='3d')
        p = ax.scatter(point[0], point[1], point[2], c=color, marker='.', alpha=0.2, edgecolors='none')
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
