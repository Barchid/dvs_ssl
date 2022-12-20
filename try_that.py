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
import random
from project.utils.barlow_transforms import BarlowTwinsTransform
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
from project.datamodules.gen1_formatted import Gen1Detection


def animate(spikes: torch.Tensor, target: dict, title="example"):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    plt.axis("off")

    for i in range(spikes.shape[0]):
        spike = spikes[i].numpy()
        frm = np.full((spike.shape[1], spike.shape[2]), 127, dtype=np.uint8)

        frm[spike[0, :, :] > 0] = 0
        frm[spike[1, :, :] > 0] = 255

        boxes = target["boxes"]
        for boxe in boxes:
            x1 = int(boxe[0])
            y1 = int(boxe[1])
            x2 = int(boxe[2])
            y2 = int(boxe[3])
            #up
            frm[y1:y1+2, x1:x2] = 0
            # left
            frm[y1:y2, x1:x1+2] = 0
            # right
            frm[y1:y2, x2:x2+2] = 0
            # bot
            frm[y2:y2+2, x1:x2] = 0
        ax.imshow(frm, cmap="Greys")  # noqa: F841
        camera.snap()

    anim = camera.animate(interval=50)
    anim.save(f"examples/{title}.gif")
    plt.close("all")


def main():
    datas = Gen1Detection(
        save_to="/datas/sandbox",
        subset="train",
        transform=BarlowTwinsTransform(
            None, timesteps=12, transforms_list=[], concat_time_channels=False
        ),
    )
    # ev, tar = datas[0]
    # _, frame, _ = ev
    # animate(frame, tar)
    # print(tar)
    
    for i in range(10):
        ev, tar = random.choice(datas)
        # print(len(ev))
        # print(tar)
        # print()
        _, frame, _ = ev
        animate(frame, tar, title=f"{i}")


if __name__ == "__main__":
    main()
