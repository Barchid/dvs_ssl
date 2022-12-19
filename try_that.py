from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tonic
from torchvision import models
import torchsummary
from plif import CIFAR10DVSNet, NMNISTNet
from project.losses.snn_loss import SnnLoss, emd_loss
from project.losses.vibcreg_loss import VIbCRegLoss
import loris
from project.utils.transform_dvs import (
    CustomToFrame,
    CutMixEvents,
    CutPasteEvent,
    ToFrame,
    get_frame_representation,
)
from project.utils.barlow_transforms import BarlowTwinsTransform
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
from project.datamodules.paf_benchmark import PAFActionRecognition
from project.datamodules.daily_action_dvs import DailyActionDVS
from project.datamodules.fer_dvs import FerDVS
import h5py

def animate(spikes: torch.Tensor):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    plt.axis("off")

    for i in range(spikes.shape[0]):
        spike = spikes[i].numpy()
        frm = np.full(
            (spike.shape[1], spike.shape[2]), 127, dtype=np.uint8
        )
        
        frm[spike[0, :, :] > 0] = 0
        frm[spike[1, :, :] > 0] = 255
        
        ax.imshow(frm, cmap="Greys")  # noqa: F841
        camera.snap()
        
    anim = camera.animate(interval=50)
    anim.save('examples/example.gif')
    plt.close('all')
    exit()
    
def main():
    pass

if __name__ == "__main__":
    main()