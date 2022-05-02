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


def main():
    pl.seed_everything(1234)
    
    tr = BarlowTwinsTransform(
        None, 30, [
            'static_rotation',
            # 'statc_translation', 
            # 'dynamic_rotation',
            # 'dynamic_translation',
            # 'cutout',
            # 'moving_occlusion'
        ]
    )
    
    datamodule = DVSDataModule(1, 'cifar10-dvs', 30, data_dir='data', barlow_transf=tr)
    
    dataloader = datamodule.train_dataloader()
    it = iter(dataloader)
    batch = next(it)
    (X, Y_a, Y_b), label = batch
    
    Y_a = Y_a[:, 0, :, :, :] # shape=(30,2,224,224) = (T,C,H,W)
    
    for t in range(30):
        frame = np.zeros((224, 224, 3))
        data = Y_a.numpy().transpose(1, 2, 0) # (C,H,W) -> (H, W, C)
        frame[:, :, 0:2] = data
        plt.imsave(f'ex_{t}.png', frame)
        
    
    

if __name__ == '__main__':
    main()