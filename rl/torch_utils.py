import torch

USE_CUDA = torch.cuda.device_count() > 0

def maybe_cuda(tensor):
    if USE_CUDA:
        return tensor.cuda()
    else:
        return tensor

def from_numpy(x):
    return maybe_cuda(torch.from_numpy(x))


def to_numpy(tensor):
    if USE_CUDA:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()
