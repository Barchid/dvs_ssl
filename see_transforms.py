from project.datamodules.cifar10dvs import CIFAR10DVS
from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import os
import random
import tonic
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from celluloid import Camera


def show_smth(tr):
    train_transform = BarlowTwinsTransform(
        CIFAR10DVS.sensor_size, timesteps=300, transforms_list=tr, concat_time_channels=False)
    dataset_train = CIFAR10DVS(save_to='data', transform=train_transform, target_transform=None)
    dataloader = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=False)

    it = iter(dataloader)
    batch = next(it)
    (X, Y_a, Y_b), label = batch

    Y_a = Y_a[0, :, :, :, :]  # shape=(100,2,224,224) = (T,C,H,W)
    X = X[0, :, :, :, :]
    fig, ax = plt.subplots()
    plt.axis("off")
    camera1 = Camera(fig)

    for t in range(300):
        frame = np.zeros((224, 224, 3))
        data = Y_a[t].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H, W, C)
        frame[:, :, 0:2] = data
        ax.imshow(frame)
        camera1.snap()

    plt.close(fig)
    print('save fig transform')
    anim = camera1.animate(interval=50)
    anim.save(f'{tr[0]}.mp4')

    fig, ax = plt.subplots()
    plt.axis("off")
    camera2 = Camera(fig)
    for t in range(300):
        frame = np.zeros((224, 224, 3))
        data = X[t].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H, W, C)
        frame[:, :, 0:2] = data
        ax.imshow(frame)
        camera2.snap()

    anim = camera2.animate(interval=50)
    print('save fig transform')
    anim.save('norm.mp4')
    plt.close(fig)
    print(label)


def main():
    pl.seed_everything(1234)

    all_tr = [
        'flip',
        'background_activity',
        'flip_polarity',
        'reverse',
        'static_rotation',
        'static_translation',
        'dynamic_rotation',
        'dynamic_translation',
        'cutout',
        'moving_occlusion'
    ]

    for tran in all_tr:
        show_smth([tran])


if __name__ == '__main__':
    main()
