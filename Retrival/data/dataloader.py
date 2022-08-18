import torch

from Retrival.data.datahelpers import collate_tuples


def getDataLoader(dataset,batch_size,workers):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=workers,
                                       pin_memory=True,#lock the memory to accerlate
                                       sampler=None,
                                       drop_last=False,#remain the less category
                                       collate_fn=collate_tuples,
                                       )
