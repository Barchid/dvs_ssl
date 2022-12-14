import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from project.models import sew_resnet, sew_resnet2
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from einops import rearrange

class SNNModule(nn.Module):
    """Some Information about SNNModule"""

    def __init__(self, model, mode="snn"):
        super(SNNModule, self).__init__()
        self.model = model
        self.mode = mode
        self.out_channels = 512

    def forward(self, x):
        if self.mode == "snn":
            functional.reset_net(self.model)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)
        elif self.mode == "3dcnn":
            x = x.permute(0, 2, 1, 3, 4)  # from (B,T,C,H,W) to (B,C,T,H,W)
        elif self.mode == "cnn" and len(x.shape) == 5:
            x = rearrange(
                x,
                "batch time channel height width -> batch (time channel) height width",
            )
        x = self.model(x)
        return x
