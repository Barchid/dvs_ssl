import torch


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def tolerance(x: torch.Tensor, y:torch.Tensor, indicator=1.0):
    # shape = (512)
    x = x.unsqueeze(0) # shape=(1,512)
    y = y.unsqueeze(0) # shape=(1,512)
    return ((x.T*y) * indicator).mean()

def uniformity(x, y, t=2):
    # shape=(512)
    return torch.linalg.norm((x-y)).pow(2).mul(-t).exp().mean().log()
