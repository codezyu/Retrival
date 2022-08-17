from Retrival.config.config import load_arg
from Retrival.util.deviceInfo import get_free_device_ids
from Retrival.model.build import buildModel
if __name__=='__main__':
    arg=vars(load_arg())
    #set model
    model=buildModel()
    #set multi gpu
    deviceId=get_free_device_ids();
    masterDevice=deviceId[0]
    model.cuda(masterDevice)
