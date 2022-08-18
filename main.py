import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

from Retrival.config.config import get_arg
from Retrival.data.dataloader import getDataLoader
from Retrival.engine.train import train
from Retrival.layer.loss import getloss
from Retrival.layer.optimizer import getoptimizer
from Retrival.layer.scheduler import getscheduler
from Retrival.util.deviceInfo import get_free_device_ids
from Retrival.model.build import buildModel
from Retrival.data.traindataset import TuplesDataset
if __name__=='__main__':
    arg=get_arg()
    #set dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainloader=getDataLoader(TuplesDataset('train',arg['trainpath'],arg['imsize'],transform=transform),
                              arg['batch_size'],
                              arg['workers'])
    validloader = getDataLoader(TuplesDataset('val', arg['valpath'], arg['imsize'], transform=transform,qsize=arg['valQsize'],poolsize=arg['valPoolsize']),arg['batch_size'],
                              arg['workers'])
    #set model
    model=buildModel()
    #set multi gpu
    deviceId=get_free_device_ids()
    masterDevice=deviceId[0]
    model.cuda(masterDevice)
    #multi
    optimizer = getoptimizer(arg, model)
    scheduler = getscheduler(arg, optimizer)
    criterion = getloss(arg).cuda()
    model = nn.DataParallel(model, device_ids=deviceId).cuda(masterDevice)
    torch.backends.cudnn.benchmark = arg['benchmark']
    #train,valid
    epoch=arg['epoch']
    np.random.seed(epoch)
    torch.manual_seed(epoch)
    torch.cuda.manual_seed_all(epoch)
    train(train_loader=trainloader,
          val_loader=validloader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          epoch=epoch,
          writer=None,
          log_interval=10)