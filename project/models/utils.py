from typing import Callable
from spikingjelly.clock_driven import surrogate, layer, neuron

import torch
import torch.nn as nn


class ConvBnSpike(nn.Sequential):
    """Convolution + BatchNorm + spiking neuron activation. Accepts input of dimension (T, B, C, H, W)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, neuron_model="LIF"):
        super(ConvBnSpike, self).__init__()
        padding = kernel_size // 2 + dilation - 1

        self.add_module('conv_bn', layer.SeqToANNContainer(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      # no bias if we use BatchNorm (because it has a BN itself)
                      bias=False,
                      dilation=dilation,
                      stride=stride),
            nn.BatchNorm2d(out_channels)
        ))

        # surrogate gradient function to use during the backward pass.
        # it is fixed here.
        surr_func = surrogate.ATan(alpha=2.0, spiking=True)

        # The spiking neuron's hyperparameters are fixed
        if neuron_model == "PLIF":
            self.add_module('spike', neuron.MultiStepParametricLIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == "IF":
            self.add_module('spike', neuron.MultiStepIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == 'LIAF':
            self.add_module('spike', MultiStepLIAFNode(nn.ReLU(), True,
                                                       detach_reset=True, surrogate_function=surr_func))
        else:
            self.add_module('spike', neuron.MultiStepLIFNode(detach_reset=True, surrogate_function=surr_func))


class ConvSpike(nn.Sequential):
    """Convolution + BatchNorm + spiking neuron activation. Accepts input of dimension (T, B, C, H, W)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=True, neuron_model="LIF"):
        super(ConvSpike, self).__init__()
        padding = kernel_size // 2 + dilation - 1

        self.add_module('conv_bn', layer.SeqToANNContainer(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=bias,  # REMEMBER : bias is not bio-plausible and hard to implement on neuromorphic hardware
                      dilation=dilation,
                      stride=stride),
            nn.BatchNorm2d(out_channels)
        ))

        # surrogate gradient function to use during the backward pass.
        # it is fixed here.
        surr_func = surrogate.ATan(alpha=2.0, spiking=True)

        # The spiking neuron's hyperparameters are fixed
        if neuron_model == "PLIF":
            self.add_module('spike', neuron.MultiStepParametricLIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == "IF":
            self.add_module('spike', neuron.MultiStepIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == 'LIAF':
            self.add_module('spike', MultiStepLIAFNode(nn.ReLU(), True,
                                                       detach_reset=True, surrogate_function=surr_func))
        else:
            self.add_module('spike', neuron.MultiStepLIFNode(detach_reset=True, surrogate_function=surr_func))


class LinearSpike(nn.Sequential):
    """FC layer + spiking neuron activation. Accepts input of dimension (T, B, C)"""

    def __init__(self, in_channels, out_channels, bias=True, neuron_model="LIF"):
        super(LinearSpike, self).__init__()

        self.add_module('fc', layer.SeqToANNContainer(
            nn.Linear(in_channels, out_channels, bias=bias)
        ))

        # surrogate gradient function to use during the backward pass.
        # it is fixed here.
        surr_func = surrogate.ATan(alpha=2.0, spiking=True)

        # The spiking neuron's hyperparameters are fixed
        if neuron_model == "PLIF":
            self.add_module('spike', neuron.MultiStepParametricLIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == "IF":
            self.add_module('spike', neuron.MultiStepIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == 'LIAF':
            self.add_module('spike', MultiStepLIAFNode(nn.ReLU(), True,
                                                       detach_reset=True, surrogate_function=surr_func))
        else:
            self.add_module('spike', neuron.MultiStepLIFNode(detach_reset=True, surrogate_function=surr_func))


class LinearBnSpike(nn.Sequential):
    """FC layer + spiking neuron activation. Accepts input of dimension (T, B, C)"""

    def __init__(self, in_channels, out_channels, bias=False, neuron_model="LIF"):
        super(LinearBnSpike, self).__init__()

        self.add_module('fc', layer.SeqToANNContainer(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.BatchNorm1d(out_channels)
        ))

        # surrogate gradient function to use during the backward pass.
        # it is fixed here.
        surr_func = surrogate.ATan(alpha=2.0, spiking=True)

        # The spiking neuron's hyperparameters are fixed
        if neuron_model == "PLIF":
            self.add_module('spike', neuron.MultiStepParametricLIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == "IF":
            self.add_module('spike', neuron.MultiStepIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == 'LIAF':
            self.add_module('spike', MultiStepLIAFNode(nn.ReLU(), True,
                                                       detach_reset=True, surrogate_function=surr_func))
        else:
            self.add_module('spike', neuron.MultiStepLIFNode(detach_reset=True, surrogate_function=surr_func))


class LIAFNode(neuron.IFNode):
    def __init__(self, act: Callable, threshold_related: bool, *args, **kwargs):
        """
        :param act: the activation function
        :type act: Callable
        :param threshold_related: whether the neuron uses threshold related (TR mode). If true, `y = act(h - v_th)`,
            otherwise `y = act(h)`
        :type threshold_related: bool
        Other parameters in `*args, **kwargs` are same with :class:`LIFNode`.
        The LIAF neuron proposed in `LIAF-Net: Leaky Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information Processing <https://arxiv.org/abs/2011.06176>`_.
        .. admonition:: Warning
            :class: warning
            The outputs of this neuron are not binary spikes.
        """
        super().__init__(*args, **kwargs)
        self.act = act
        self.threshold_related = threshold_related

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        if self.threshold_related:
            y = self.act(self.v - self.v_threshold)
        else:
            y = self.act(self.v)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return y


class MultiStepLIAFNode(LIAFNode):
    def __init__(self, act: Callable, threshold_related: bool, *args, **kwargs):
        super().__init__(act, threshold_related, *args, **kwargs)
        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq  # analogous spikes

    def extra_repr(self):
        return super().extra_repr() + f', backend=torch'


class MeanSpike(nn.Module):
    def __init__(self):
        super(MeanSpike, self).__init__()

    def forward(self, x):
        # shape = (T,B,C) to (B,C)
        return torch.mean(x, dim=0)

class BNTT(nn.Module):
    """Some Information about BNTT"""
    def __init__(self, num_features: int, timesteps: int):
        super(BNTT, self).__init__()
        self.timesteps = timesteps
        self.num_features = num_features
        self.bntt = nn.ModuleList([nn.BatchNorm2d(self.num_features, eps=1e-4, momentum=0.1) for i in range(self.timesteps)])
        
        # Turn off bias of BNTT
        # for bn_temp in self.bntt:
        #     bn_temp.bias = None
        
    def forward(self, x):
        # x.shape = (T,B,C,H,W)
        out = []
        for t in range(x.shape[0]):
            print(x.shape, t)
            x_t = x[t]
            bn = self.bntt[t]
            x_t = bn(x_t)
            out.append(x_t)
            
        for t in range(len(out)):
            out[t] = out[t].unsqueeze(0)
            
        return torch.cat(out, 0)