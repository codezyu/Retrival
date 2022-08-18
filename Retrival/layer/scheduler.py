import math

import torch


def getscheduler(arg,optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=arg['gamma'])