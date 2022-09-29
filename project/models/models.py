import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


def get_encoder(in_channels: int) -> nn.Module:
    resnet18 = models.resnet18(progress=True)

    resnet18.fc = nn.Identity()

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

def get_encoder_3d(in_channels: int) -> nn.Module:
    # resnet18 = models.video.r3d_18()
    # resnet18.fc = nn.Identity()
    # resnet18.stem[0] = nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3))
    resnet18 = models.video.r2plus1d_18()
    resnet18.fc = nn.Identity()
    resnet18.stem[0] = nn.Conv3d(in_channels=in_channels, out_channels=45, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
    return resnet18


# def get_encoder(in_channels: int):
#     encoder = models.resnet18(pretrained=False)
#     encoder.classifier = nn.Identity()  # remove the final classification layer/module

#     if in_channels != 3:
#         encoder.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=(3, 3),
#                                            stride=(2, 2), padding=(1, 1), bias=False)

#     return encoder


def get_projector(in_channels: int = 512):
    projector = nn.Sequential(
        nn.Linear(in_channels, 3 * in_channels),
        nn.BatchNorm1d(3 * in_channels),
        nn.ReLU(),
        nn.Linear(3 * in_channels, 3 * in_channels),
        nn.BatchNorm1d(3 * in_channels),
        nn.ReLU(),
        nn.Linear(3 * in_channels, 3 * in_channels),
    )

    return projector
