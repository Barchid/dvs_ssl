import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from project.models import sew_resnet
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

from project.models.utils import (
    LinearBnSpike,
    LinearSpike,
    MeanSpike,
    MultiStepLIAFNode,
)


def get_projector_lif(in_channels=512, neuron_model: str = "IF") -> nn.Sequential:
    projector = nn.Sequential(
        LinearBnSpike(in_channels, 3 * in_channels, neuron_model=neuron_model),
        LinearBnSpike(3 * in_channels, 3 * in_channels, neuron_model=neuron_model),
        LinearSpike(3 * in_channels, 3 * in_channels, neuron_model=neuron_model),
        # MeanSpike(),
    )

    return projector


def get_encoder_snn(in_channels: int, T: int, output_all: bool):
    resnet18 = sew_resnet.MultiStepSEWResNet(
        block=sew_resnet.MultiStepBasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=True,
        T=T,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepIFNode,
        detach_reset=True,
        surrogate_function=surrogate.ATan(),
        output_all=output_all,
    )

    # resnet18.layer4[-1].sn2 = MultiStepLIAFNode(
    #     torch.nn.ReLU(),
    #     threshold_related=False,
    #     detach_reset=True,
    #     surrogate_function=surrogate.ATan(),
    # )

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet18


class ProjectorSSL(nn.Module):
    def __init__(self, in_channels: int = 512):
        super(ProjectorSSL, self).__init__()
        self.l1 = LinearBnSpike(in_channels, 3 * in_channels, neuron_model="IF")
        self.l2 = LinearBnSpike(in_channels * 3, 3 * in_channels, neuron_model="IF")

        self.l3_linear = layer.SeqToANNContainer(
            nn.Linear(in_channels * 3, in_channels * 3, bias=True)
        )
        self.l3_spike = neuron.MultiStepIFNode(
            detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=2.0, spiking=True),
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        lin = self.l3_linear(x)
        x = self.l3_spike(lin)
        return x, lin
