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
    pl.seed_everything(2222)

    tr = [
        'static_rotation',
        # 'statc_translation',
        # 'dynamic_rotation',
        # 'dynamic_translation',
        # 'cutout',
        # 'moving_occlusion'
    ]

    datamodule = DVSDataModule(1, 'cifar10-dvs', 30, data_dir='data', barlow_transf=tr, mode="snn")
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = datamodule.train_dataloader()
    it = iter(dataloader)
    batch = next(it)
    (X, Y_a, Y_b), label = batch

    Y_a = Y_a[0, :, :, :, :]  # shape=(30,2,224,224) = (T,C,H,W)

    fig1 = plt.figure()
    plt.axis("off")
    camera1 = Camera(fig1)
    
    for t in range(30):
        frame = np.zeros((224, 224, 3))
        data = Y_a[t].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H, W, C)
        frame[:, :, 0:2] = data
        plt.imshow(frame)
        camera1.snap()
        
    plt.close()
    
    anim = camera1.animate(interval=200)
    anim.save('trans.mp4')
        
    fig2 = plt.figure()
    plt.axis("off")
    camera2 = Camera(fig2)
    for t in range(30):
        frame = np.zeros((224, 224, 3))
        data = X[t].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H, W, C)
        frame[:, :, 0:2] = data
        plt.imshow(frame)
        camera2.snap()
        
    anim = camera2.animate(interval=200)
    anim.save('trans.mp4')

    print(label)


if __name__ == '__main__':
    main()
