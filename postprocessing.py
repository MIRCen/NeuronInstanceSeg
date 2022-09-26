# -*- coding: utf-8 -*-
import numpy as np
import timeit
import cv2
from skimage import measure
from skimage.morphology import erosion, disk, dilation, erosion
import argparse
import timeit
from skimage.morphology.greyreconstruct import reconstruction
from skimage.segmentation import watershed
from skimage.color import label2rgb

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input mask image path')
parser.add_argument('-o', help='output label image path')
parser.add_argument('-r', help='ultimate erosion disk radius')
args = parser.parse_args()

pred = cv2.imread('%s' %(str(args.i)), cv2.IMREAD_UNCHANGED)

pred_msk = np.zeros_like(pred[..., 0], dtype='uint16') 
pred_msk = np.where((pred[..., 0] > pred[..., 1]) & (pred[..., 0] > pred[..., 2]), 1, pred_msk)

pred_msk = pred_msk.astype(np.uint16)

t = t0 = timeit.default_timer()
# ultimate erosion based on geodesic reconstruction
print('**************ultimate erosion***************')
nb_iter = 0

r = 10
if args.r:
    r = int(args.r)

img = pred_msk
img_niter = np.zeros_like(pred_msk, dtype='uint16') 
while(np.max(img) > 0):
    img_ero = erosion(img, disk(r))
    nb_iter = nb_iter+1
    reconst= reconstruction(img_ero, img,'dilation')
    residues = img - reconst
    img_niter = np.where(residues==1, nb_iter, img_niter)
    img = img_ero

print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))
t = timeit.default_timer()

# residues relabel
print('**************relabel residues***************')
img_residue = img_niter
img_residue[img_residue>0] = 1
img_residue = dilation(img_residue, disk(3))
img_residue = measure.label(img_residue, connectivity=2, background=0)
img_residue = erosion(img_residue, disk(3))

print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))
t = timeit.default_timer()

# dynamic reconstruction
print('**************dynamic reconstruction***************')
img_rc = np.zeros_like(img_residue, dtype='uint16') 
i = np.max(img_niter)

while i > 1:
    img_rc = np.where(img_niter == i, img, img_rc) 
    img_rc = dilation(img_rc, disk(r))
    i = i-1

img_rc = np.where(img_niter == i, img_residue, img_rc) 

print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))
t = timeit.default_timer()

# apply watershed
print('**************watershed reconstruction***************')
nucl_msk = (255 - pred[..., 0])
mask = np.zeros_like(pred[..., 0], dtype='uint16')
mask = np.where(((pred[..., 0] + pred[..., 1]) > pred[..., 2]), 1, mask)

y_pred = watershed(nucl_msk, img_rc, mask=mask, watershed_line=False)
y_pred = measure.label(y_pred, connectivity=1, background=0)

props = measure.regionprops(y_pred)

print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))
t = timeit.default_timer()

# remove small objects and keep only disk-like objects
print('**************remove small and non-circular object***************')

for i in range(len(props)):
    # if props[i].area < 200:
    if props[i].area < 128:
        y_pred[y_pred == i+1] = 0
    if props[i].eccentricity > 0.97:       
        y_pred[y_pred == i+1] = 0

pred_labels = measure.label(y_pred, connectivity=1, background=0)
pred_labels = pred_labels.astype('uint16')

cv2.imwrite('%s' %(str(args.o)), pred_labels)

# color labels
clr_labels = label2rgb(pred_labels, bg_label=0)
clr_labels *= 255
clr_labels = clr_labels.astype('uint8')
cv2.imwrite('%s' %(str(args.o).replace('.tif', '.png')), clr_labels)

print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))

elapsed = timeit.default_timer() - t0
print('Total time: {:.3f} s'.format(elapsed))