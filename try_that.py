from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tonic
from torchvision import models
from project.utils.transform_dvs import (
    CustomToFrame,
    CutMixEvents,
    CutPasteEvent,
    ToFrame,
    get_frame_representation,
)
import cv2
from project.utils.barlow_transforms import BarlowTwinsTransform
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
from project.datamodules.gen1_formatted import Gen1Detection

def animate(spikes: torch.Tensor, target: dict):
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
        
        boxes = target["boxes"]
        for boxe in boxes:
            x1 = int((boxe[0]/304)*128)
            y1 = int((boxe[1]/240)*128)
            x2 = int((boxe[2]/304)*128)
            y2 = int((boxe[3]/240)*128)
            frm[y1:y2, x1:x2] = 0
        ax.imshow(frm, cmap="Greys")  # noqa: F841
        camera.snap()
        
    anim = camera.animate(interval=50)
    anim.save('examples/example.gif')
    plt.close('all')
    
def main():
    datas = Gen1Detection(save_to="/datas/sandbox", subset="train", transform=BarlowTwinsTransform(None, timesteps=50, transforms_list=[], concat_time_channels=False))
    print(len(datas))
    ev, tar = datas[0]
    _, frame, _ = ev
    animate(frame, tar)
    print(tar)

if __name__ == "__main__":
    main()