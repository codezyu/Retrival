import torch

def getoptimizer(arg):
    return torch.optim.Adam(lr=arg['lr'],weight_decay=arg['weight_decay'])