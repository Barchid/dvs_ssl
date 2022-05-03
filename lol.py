import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt

def main():
    datamodule = DVSDataModule(32, 'cifar10-dvs', 32, data_dir='data', barlow_transf=[])
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    
    i = 0
    for batch in dataloader:
        print(f"number : {i} / {len(dataloader)}")
        i += 1
    

if __name__ == "__main__":
    main()