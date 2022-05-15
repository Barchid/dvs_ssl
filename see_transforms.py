from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import tonic
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from celluloid import Camera

def main():
    pl.seed_everything(4)

    tr = [
        'flip'
        # 'static_rotation',
        # 'static_translation',
        # 'dynamic_rotation',
        # 'dynamic_translation',
        # 'cutout',
        # 'moving_occlusion'
    ]

    datamodule = DVSDataModule(1, 'cifar10-dvs', 300, data_dir='data', barlow_transf=tr, mode="snn")
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()
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
    
    anim = camera1.animate(interval=50)
    anim.save('trans.webm')
        
    fig, ax = plt.subplots()
    plt.axis("off")
    camera2 = Camera(fig)
    for t in range(100):
        frame = np.zeros((224, 224, 3))
        data = X[t].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H, W, C)
        frame[:, :, 0:2] = data
        ax.imshow(frame)
        camera2.snap()
        
    anim = camera2.animate(interval=50)
    anim.save('norm.webm')
    plt.close(fig)
    print(label)


if __name__ == '__main__':
    main()
