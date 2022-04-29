import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# FROM https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/barlow-twins.html#Barlow-Twins-Loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, Z_a, Z_b):
        # normalize repr. along the batch dimension
        Z_a = (Z_a - Z_a.mean(0)) / Z_a.std(0)
        Z_b = (Z_b - Z_b.mean(0)) / Z_b.std(0)
        
        # N x D, where N is the batch size and D is output dim of projection head
        Z_a = (Z_a - torch.mean(Z_a, dim=0)) / torch.std(Z_a, dim=0)
        Z_b = (Z_b - torch.mean(Z_b, dim=0)) / torch.std(Z_b, dim=0)

        cross_corr = torch.matmul(Z_a.T, Z_b) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
