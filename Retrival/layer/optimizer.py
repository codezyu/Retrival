import torch
def getoptimizer(arg,model):
    return torch.optim.Adam(model.net.parameters(),lr=arg['lr'],weight_decay=arg['weight_decay'])