import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def emd_loss(a: torch.Tensor, b: torch.Tensor):
    """Implementation of the efficient way of calculating the restricted EMD distance
    (shown in paper https://www.frontiersin.org/articles/10.3389/fncom.2019.00082/full)

    Args:
        a (torch.Tensor): dim=(T,B,C) the spike train a
        b (torch.Tensor): dim=(T,B,C) the spike train b
    """
    # normalize spike trains over spike numbers
    n_a = torch.count_nonzero(a, dim=0)
    a = a / (n_a + 1e-5)
    n_b = torch.count_nonzero(b, dim=0)
    b = b / (n_b + 1e-5)
    
    # cumsum
    cum_a = torch.cumsum(a, 0) # cumulative function of spike train a
    cum_b = torch.cumsum(b, 0) # cumulative function of spike train b
    
    # |elementiwe difference|
    diff = torch.abs(cum_a - cum_b)
    
    # sum is the result
    return diff.sum()

def 
