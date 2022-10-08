'''
Plots a sample from given dataset.

Usage examples:
$ python3 plotsample.py SemanticKITTI 0
$ python3 plotsample.py SemanticKITTI 0 --topview
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['Sydney', 'SemanticKITTI', 'ICCV17ShapeNetSeg', 'ICCV17ShapeNetClass', 'EricyiShapeNetSeg', 'EricyiShapeNetClass', 'Stanford3d'])
parser.add_argument('i', type=int)
parser.add_argument('--datadir', default='/data')
parser.add_argument('--topview', action='store_true')
args = parser.parse_args()

import pnets as pn

ds = getattr(pn.data, args.dataset)
ds = ds(args.datadir, 'test', None)

P, Y = ds[args.i]
f = pn.plot.plot_topview if args.topview else pn.plot.plot3d
c = Y if ds.segmentation else 'k'
ax = f(P, c)
pn.plot.zoomin(ax, P)
if not ds.segmentation:
    ax.set_title(f'class={Y}')
pn.plot.show()

print(len(ds))
