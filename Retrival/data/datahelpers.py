import os
from PIL import Image

import torch
import numpy as np
def default_loader(path):
    #return numpy
    return np.load(path.split('.j')[0]+'.npy',allow_pickle=True)

def imresize(img, imsize):
    #eg: 224 224 3
    img=img.resize((imsize, imsize))
    return img

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]