import torch
import numpy as np
from torchvision.models import ResNet
from torchvision.models.video.resnet import VideoResNet
from project.models.sew_resnet import MultiStepSEWResNet
from spikingjelly.clock_driven import functional

def forward_analyze_cnn(encoder: ResNet, x: torch.Tensor):
    # See note [TorchScript super()]
    x = encoder.conv1(x)
    x = encoder.bn1(x)
    stem_feat = encoder.relu(x)
    
    x = encoder.maxpool(stem_feat)

    res2_feat = encoder.layer1(x)
    res3_feat = encoder.layer2(res2_feat)
    res4_feat = encoder.layer3(res3_feat)
    x = encoder.layer4(res4_feat)

    x = encoder.avgpool(x)
    x = torch.flatten(x, 1)
    x = encoder.fc(x)

    return x, stem_feat, res2_feat, res3_feat, res4_feat

def forward_analyze_snn(encoder: MultiStepSEWResNet, x: torch.Tensor):
    # See note [TorchScript super()]
    x_seq = None
    if x.dim() == 5:
        # x.shape = [T, N, C, H, W]
        x_seq = functional.seq_to_ann_forward(x, [encoder.conv1, encoder.bn1])
    else:
        assert (
            encoder.T is not None
        ), "When x.shape is [N, C, H, W], encoder.T can not be None."
        # x.shape = [N, C, H, W]
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x.unsqueeze_(0)
        x_seq = x.repeat(encoder.T, 1, 1, 1, 1)

    stem_feat = encoder.sn1(x_seq)

    x_seq = functional.seq_to_ann_forward(stem_feat, encoder.maxpool)

    res2_feat = encoder.layer1(x_seq)

    res3_feat = encoder.layer2(res2_feat)

    res4_feat = encoder.layer3(res3_feat)

    x_seq = encoder.layer4(res4_feat)

    # if not encoder.output_all:
    x_seq = functional.seq_to_ann_forward(x_seq, encoder.avgpool)

    x_seq = torch.flatten(x_seq, 2)
    # x_seq = encoder.fc(x_seq.mean(0))
    x_seq = functional.seq_to_ann_forward(x_seq, encoder.fc)
    x_seq = encoder.final_neurons(x_seq)

    if encoder.output_all:
        return x_seq
    else:
        # encoder.final_neurons(x_seq)
        return encoder.final_neurons.v_seq[-1], stem_feat.clone().mean(0), res2_feat.clone().mean(0), res3_feat.clone().mean(0), res4_feat.clone().mean(0)
        # return torch.mean(x_seq, dim=0)  # mean value of all analog spikes (at each time-step)

def forward_analyze_3dcnn(encoder: VideoResNet, x: torch.Tensor):
    H, W = x.shape[-2], x.shape[-1]
    stem_feat = encoder.stem(x)
    
    res2_feat = encoder.layer1(stem_feat)
    res3_feat = encoder.layer2(res2_feat)
    res4_feat = encoder.layer3(res3_feat)
    res5_feat = encoder.layer4(res4_feat)

    x = encoder.avgpool(res5_feat)
    # Flatten the layer to fc
    x = x.flatten(1)
    x = encoder.fc(x)
    
    stem_feat = stem_feat.clone().mean(2)
    res2_feat = torch.nn.functional.interpolate(res2_feat.clone().mean(2), size=(H//4, W//4))
    res3_feat = torch.nn.functional.interpolate(res3_feat.clone().mean(2), size=(H//8, W//8))
    res4_feat = torch.nn.functional.interpolate(res4_feat.clone().mean(2), size=(H//16, W//16))
    
    return x, stem_feat, res2_feat, res3_feat, res4_feat