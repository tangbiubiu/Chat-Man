from torch.cuda import empty_cache, ipc_collect, device, is_available

def choose_device():
    '''
    选择设备
    不考虑CPU，CUDA不可用时抛出异常
    '''
    if is_available():
        device = "cuda"
        deivce_id = "0"
    else:
        raise Exception("CUDA不可用！")
    cuda_device = f"{device}:{deivce_id}" if deivce_id else device
    return cuda_device

# 清理GPU内存函数
def torch_gc(CUDA_DEVICE):
    if is_available():
        with device(CUDA_DEVICE): 
            empty_cache()  # 清空CUDA缓存
            ipc_collect()  # 收集CUDA内存碎片