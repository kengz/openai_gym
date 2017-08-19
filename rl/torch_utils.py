import torch

def from_numpy(x):
    return torch.from_numpy(x).cuda()

def to_numpy(tensor):
    return tensor.cpu().numpy()

def maybe_cuda(tensor):
    return tensor.cuda()
