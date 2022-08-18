import torch
from torch import nn
def triplet_loss(x, label, margin=0.1):
    # x is D x N
    dim = x.size(0)  # D
    nq = torch.sum(label.data == -1).item()  # number of tuples
    S = x.size(1) // nq  # number of images per tuple including query: 1+1+n

    xa = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, dim).permute(1, 0)
    xp = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, dim).permute(1, 0)
    xn = x[:, label.data == 0]

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0))

class TripletLoss(nn.Module):

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, label):
        return triplet_loss(x, label, margin=self.margin)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'
def getloss(arg):
    return TripletLoss(arg['loss_margin'])