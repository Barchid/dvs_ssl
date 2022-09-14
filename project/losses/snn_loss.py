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
    summation = torch.sum(diff, dim=0)
    
    # sum is the result
    return torch.mean(summation)




class SnnLoss(nn.Module):

    def __init__(self, invariance_loss_weight: float = 25., variance_loss_weight: float = 25., covariance_loss_weight: float = 1., invariance_mode="emd", variance_mode="n", covariance_mode = "n", multiple_proj: bool=False):
        super(SnnLoss, self).__init__()
        
        self.invariance_loss_weight = invariance_loss_weight
        self.variance_loss_weight = variance_loss_weight
        self.covariance_loss_weight = covariance_loss_weight
        self.multiple_proj = multiple_proj
        
        
        if invariance_mode == "emd":
            self.invariance_loss = emd_loss 
        elif invariance_mode == "mse":
            self.invariance_loss = F.mse_loss
        elif invariance_mode == "smoothl1":
            self.invariance_loss = F.smooth_l1_loss
        
        self.invariance_mode = invariance_mode
        self.variance_mode = variance_mode
        self.covariance_mode = covariance_mode

    def forward(self, Z_a, Z_b):
        if self.multiple_proj: # if this flag is True, it means that the membrane potentials are also included in the input
            Z_a, V_a = Z_a
            Z_b, V_b = Z_b
            
        assert Z_a.shape == Z_b.shape and len(Z_a.shape) == 3 # ensure (T,B,C)
        assert V_a.shape == V_b.shape and len(V_a.shape) == 3 # ensure (T,B,C)

        # invariance loss
        loss_inv = self.invariance_loss(Z_a, Z_b)

        if self.multiple_proj:
            T = V_a.shape[0]
            # use Z_* as V_* because it is easier to code and I'm tired to code properly
            Z_a = torch.mean(V_a, dim=0) #/ T # number of spikes
            Z_b = torch.mean(V_b, dim=0) #/ T # same 
        else:
            T = Z_a.shape[0]
            Z_a = torch.mean(Z_a, dim=0) #/ T # number of spikes
            Z_b = torch.mean(Z_b, dim=0) #/ T # same 
        # print(Z_a, Z_b)

        # variance loss
        std_Z_a = torch.sqrt(Z_a.var(dim=0) + 1e-04)
        std_Z_b = torch.sqrt(Z_b.var(dim=0) + 1e-04)
        loss_v_a = torch.mean(F.relu(1 - std_Z_a))
        loss_v_b = torch.mean(F.relu(1 - std_Z_b))
        loss_var = loss_v_a + loss_v_b
        
        # print('loss_var=', loss_var)

        # covariance loss
        N, D = Z_a.shape
        Z_a = Z_a - Z_a.mean(dim=0)
        Z_b = Z_b - Z_b.mean(dim=0)
        cov_Z_a = ((Z_a.T @ Z_a) / (N - 1)).square()  # DxD
        cov_Z_b = ((Z_b.T @ Z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_Z_a.sum() - cov_Z_a.diagonal().sum()) / D
        loss_c_b = (cov_Z_b.sum() - cov_Z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b
        
        # print('loss_cov=', loss_cov)

        weighted_inv = loss_inv * self.invariance_loss_weight
        weighted_var = loss_var * self.variance_loss_weight
        weighted_cov = loss_cov * self.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return loss
    