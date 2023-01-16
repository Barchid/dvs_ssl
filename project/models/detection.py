import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from project.models import sew_resnet, sew_resnet2
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from einops import rearrange
from project.models import densenet


def get_densenet_encoder(in_channels: int, timesteps: int):
    encoder = densenet.MultiStepSpikingDenseNet(
        growth_rate=24,
        num_init_channels=in_channels,
        norm_layer=nn.BatchNorm2d,
        T=timesteps,
        neuron_fun=neuron.MultiStepIFNode
    )
    return encoder

# def get_densenet_cnn(in_channels: int, timesteps: int)


class SNNModule(nn.Module):
    """Some Information about SNNModule"""

    def __init__(self, model, mode="snn"):
        super(SNNModule, self).__init__()
        self.model = model
        self.mode = mode
        self.out_channels = 512
        self._adapt()

    def forward(self, x):
        if self.mode == "snn":
            functional.reset_net(self.model)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)
            return self._forward_snn(x)
        elif self.mode == "3dcnn":
            x = x.permute(0, 2, 1, 3, 4)  # from (B,T,C,H,W) to (B,C,T,H,W)
            return self._forward_3dcnn(x)
        elif self.mode == "cnn" and len(x.shape) == 5:
            x = rearrange(
                x,
                "batch time channel height width -> batch (time channel) height width",
            )
            x = self._forward_cnn(x)
            return x
        elif self.mode == "cnn":
            x = self._forward_cnn(x)
            return x

        x = self.model(x)
        return x

    def _adapt(self):
        if self.mode == "snn":
            self.model.avgpool = nn.Identity()
            self.model.fc = nn.Identity()
            self.model.final_neurons = nn.Identity()
        elif self.mode == "cnn":
            self.model.avgpool = nn.Identity()
            self.model.fc = nn.Identity()
        else:  # 3dcnn
            self.model.avgpool = nn.AdaptiveAvgPool3d((1, 4, 4))
            self.model.fc = nn.Identity()

    def _forward_cnn(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

    def _forward_3dcnn(self, x):
        x = self.model.stem(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x.squeeze()

    def _forward_snn(self, x):
        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            x_seq = functional.seq_to_ann_forward(x, [self.model.conv1, self.model.bn1])
        else:
            assert (
                self.model.T is not None
            ), "When x.shape is [N, C, H, W], self.model.T can not be None."
            # x.shape = [N, C, H, W]
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x.unsqueeze_(0)
            x_seq = x.repeat(self.model.T, 1, 1, 1, 1)

        x_seq = self.model.sn1(x_seq)

        x_seq = functional.seq_to_ann_forward(x_seq, self.model.maxpool)

        x_seq = self.model.layer1(x_seq)

        x_seq = self.model.layer2(x_seq)

        x_seq = self.model.layer3(x_seq)

        x_seq = self.model.layer4(x_seq)
        return torch.mean(x_seq, 0)
