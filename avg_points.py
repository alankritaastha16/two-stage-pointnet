'''
Average #points for a given dataset.
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--datadir', default='/data')
args = parser.parse_args()

from tqdm import tqdm
import pnets as pn
import numpy as np

ds = getattr(pn.data, args.dataset)
ds = ds(args.datadir, 'train', None)

points = [len(S) for P, S in tqdm(ds)]
avg = np.mean(points)
outlier_quantile = np.quantile(points, 0.6)

print(args.dataset, avg, outlier_quantile)
