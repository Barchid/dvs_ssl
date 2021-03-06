from tkinter import N
from project.datamodules.cifar10dvs import CIFAR10DVS
from project.datamodules.dvs_datamodule import DVSDataModule
from project.datamodules.ncaltech101 import NCALTECH101
from project.datamodules.ncars import NCARS
from project.utils.barlow_transforms import BarlowTwinsTransform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import os
import random
import tonic
from tonic.datasets import DVSGesture
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from celluloid import Camera

lolil = 0

def show_smth(tr):
    global lolil
    
    train_transform = BarlowTwinsTransform(
        tonic.datasets.NMNIST.sensor_size, timesteps=100, transforms_list=tr, concat_time_channels=False)
    # dataset_train = NCARS(save_to='data', transform=None, target_transform=None)
    # ev, _ = random.choice(dataset_train)
    # print(ev.shape)
    # exit()
    dataset_train = tonic.datasets.NMNIST(save_to='data', transform=train_transform, target_transform=None)
    dataloader = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=True)

    it = iter(dataloader)
    batch = next(it)
    (X, Y_a, Y_b), label = batch
    print(X.shape)
    Y_a = Y_a[0, :, :, :, :]  # shape=(100,2,224,224) = (T,C,H,W)
    T = Y_a.shape[0]
    X = X[0, :, :, :, :]
    fig, ax = plt.subplots()
    plt.axis("off")
    camera1 = Camera(fig)

    for t in range(T):
        frame = np.zeros((224, 224, 3))
        data = Y_a[t].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H, W, C)
        frame[:, :, 0:2] = data
        ax.imshow(frame)
        camera1.snap()

    plt.close(fig)
    print('save fig transform', tr)
    anim = camera1.animate(interval=40)
    anim.save(f'examples/{lolil}.mp4')
    lolil += 1

    if lolil == 1:
        print('norm now')
        fig, ax = plt.subplots()
        plt.axis("off")
        camera2 = Camera(fig)
        for t in range(T):
            frame = np.zeros((224, 224, 3))
            data = X[t].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H, W, C)
            frame[:, :, 0:2] = data
            ax.imshow(frame)
            camera2.snap()

        anim = camera2.animate(interval=40)
        print('save fig transform')
        anim.save('norm.mp4')
        plt.close(fig)
        print(label)


def main():
    # pl.seed_everything(4)
    os.makedirs('examples', exist_ok=True)
    all_tr = [
        'flip',
        'background_activity',
        'flip_polarity',
        'reverse',
        # 'static_rotation',
        # 'static_translation',
        'dynamic_rotation',
        'dynamic_translation',
        # 'cutout',
        # 'crop',
        'cutpaste',
        'moving_occlusion'
    ]
    
    # show_smth([])
    # exit()
    for _ in range(1):
        show_smth(all_tr)
        print('next')


if __name__ == '__main__':
    main()
