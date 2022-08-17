from torch.testing._internal.common_quantization import AverageMeter


def train(train_loader, model, criterion, optimizer, epoch,writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    