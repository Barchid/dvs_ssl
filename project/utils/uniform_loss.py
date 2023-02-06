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


def uniformity_orig(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def tolerance_orig(x, l):
    total_distances = 0.0
    for i in range(int(l.min()), int(l.max()) + 1):
        cur_features = x[(l == i).nonzero(as_tuple=True)[0]]
        distances = torch.mm(cur_features, cur_features.T)
        mask = torch.ones((cur_features.shape[0], cur_features.shape[0])) - torch.eye(cur_features.shape[0])
        masked_distances = distances * mask
        total_distances += masked_distances.mean()
    return total_distances.mean() / (1 + l.max() - l.min())