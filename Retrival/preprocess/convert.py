import os
import sys

import cv2
import numpy as np


# convert jpg to numpy array
def jpg2npy(paths):
    for dirs in paths:
        for dir in os.listdir(dirs):
            one_dir = '/'.join([dirs, dir])
            for qpimg in os.listdir(one_dir):
                if not qpimg.endswith('jpg'):
                    continue
                img_dir = '/'.join([one_dir,qpimg])
                im1 = cv2.imread(img_dir)
                im2 = np.array(im1)
                np.save(img_dir.split('.j')[0] + '.npy', im2)
if __name__=='__main__':
    for i in range(1,len(sys.argv)):
        dirs=sys.argv[i]
        for dir in os.listdir(dirs):
            one_dir = '/'.join([dirs, dir])
            for qpimg in os.listdir(one_dir):
                if not qpimg.endswith('jpg'):
                    continue
                img_dir = '/'.join([one_dir,qpimg])
                im1 = cv2.imread(img_dir)
                im2 = np.array(im1)
                np.save(img_dir.split('.j')[0] + '.npy', im2)