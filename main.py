from torch import nn
from torchvision.transforms import transforms

from Retrival.config.config import get_arg
from Retrival.data.dataloader import getDataLoader
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
                              arg['batch_size'])
    validloader = getDataLoader(TuplesDataset('val', arg['valpath'], arg['imsize'], transform=transform),)
    #set model
    model=buildModel()
    #set multi gpu
    deviceId=get_free_device_ids()
    masterDevice=deviceId[0]
    model.cuda(masterDevice)
    #multi
    model = nn.DataParallel(model, device_ids=deviceId).cuda(masterDevice)
    #train

    #valid