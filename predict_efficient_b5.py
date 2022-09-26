from os import path, mkdir, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import timeit
import cv2
from tqdm import tqdm
from efficientunet import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input test image folder', type=str)
parser.add_argument('-model', help='pretrained model weight path', type=str)
parser.add_argument('-o', help='output probability map folder', type=str)
args = parser.parse_args()

test_folder = args.i
models_folder = args.model
test_pred = args.o

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127.5
    x -= 1.
    return x

def bgr_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(17, 17))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return lab[..., np.newaxis]

if __name__ == '__main__':
    

    if not path.isdir(test_pred):
        mkdir(test_pred)
        
    print('Loading models')

    # fold_nums = [0, 1, 2, 3]
    fold_nums = [1]
    for it in range(4):
        if it not in fold_nums:
            continue
        # image size depends on the GPU memory
        model = get_efficient_unet_b5((1344, 1344, 3), pretrained=False, block_type='transpose', concat_input=True)
        model.load_weights(path.join(models_folder, 'efficient_b5_weights_{0}.h5'.format(it)))
        
    print('Predicting test')
    t0 = timeit.default_timer()
    for root, dirs, files in os.walk(test_folder, topdown=False):
        for d in tqdm(files):
            final_mask = None
            fid = d
            img = cv2.imread(path.join(test_folder, '{0}'.format(fid)), cv2.IMREAD_COLOR)
            if final_mask is None:
                final_mask = np.zeros((img.shape[0], img.shape[1], 3))
            inp = []
            inp.append(img)
            inp = np.asarray(inp)
            inp = preprocess_inputs(inp)
            pred = model.predict(inp)

            final_mask = pred[0]
            final_mask = final_mask * 255
            final_mask = final_mask.astype('uint8')
            cv2.imwrite(path.join(test_pred, fid), final_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
    elapsed = timeit.default_timer() - t0
    print('Time: {:.6f} s'.format(elapsed))