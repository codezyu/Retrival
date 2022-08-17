import torch.nn as nn
from torchvision.models import efficientnet_v2_l
from Retrival.layer.pool import GeneralizedMeanPooling
#ouput 1280
class backboneModel(nn.Module):
    def __init__(self):
        super(backboneModel, self).__init__()
        self.net=efficientnet_v2_l()
        self.net.avgpool=GeneralizedMeanPooling()
        self.net.classifier=nn.Sequential()
    def forward(self, x):
        o = self.net(x)