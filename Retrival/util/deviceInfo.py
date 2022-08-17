import pynvml
def get_free_device_ids():
    pynvml.nvmlInit()
    num_device = pynvml.nvmlDeviceGetCount()
    free_device_id = []
    for i in range(num_device):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        men_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(men_info.total,men_info.free)
        # import pdb; pdb.set_trace()
        if men_info.free >= men_info.total*0.99:
            free_device_id.append(i)
    return free_device_id