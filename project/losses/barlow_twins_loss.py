import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        N = Z_a.size(0)
        D = Z_a.size(1)

        # empirical cross-correlation matrix
        c = torch.mm(Z_a.T, Z_b) / N

        c_diff = (c - torch.eye(D, device=device)).pow(2)  # "on-diagonal" term
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_coeff  # "off-diagonal" term

        return c_diff.sum()
