from argparse import ArgumentParser


def load_arg():
    parser = ArgumentParser(description="Pytorch Training")
    parser.add_argument("-config_file", "--CONFIG_FILE", type=str, required=False, help="Path to config file")
    parser.add_argument("-tag", "--TAG", type=str)
    parser.add_argument("-max_num_devices", type=int)
    # DATA
    parser.add_argument("-train_num_workers", "--DATA.DATALOADER.TRAIN_NUM_WORKERS", type=int,
                        help='Num of data loading threads. ')
    # MODEL
    parser.add_argument("-load_path", type=str,
                        help='Path of pretrained model. ')
    parser.add_argument("-device", type=int, nargs='+',
                        help="list of device_id, e.g. [0,1,2]")

    arg = parser.parse_args()
    return arg
def get_arg():
    return {
        'trainpath':'1',
        'valpath':'2',
        'imsize':224,
        'batch_size':128,
        'workers':8,
        'valQsize':float('Inf'),
        'valPoolsize':float('Inf'),
        'benchmark':True,
        'lr':3e-4,
        'weight_decay':1e-5
    }