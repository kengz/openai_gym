import torch

def from_numpy(x):
    x = torch.from_numpy(x) 
    if torch.cuda.device_count() > 0:
        return x.cuda()
    else:
        return x


def to_numpy(tensor):
    if torch.cuda.device_count() > 0:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()

def maybe_cuda(tensor):
    if torch.cuda.device_count() > 0:
        return tensor.cuda()
    else:
        return tensor
