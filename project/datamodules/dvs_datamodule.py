from typing import Optional
from cv2 import transform
import pytorch_lightning as pl
import torch
from torch import save
from torch.utils import data
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
import tonic
import os
import numpy as np

from project.datamodules.cifar10dvs import CIFAR10DVS
from project.datamodules.ncars import NCARS
from project.utils.barlow_transforms import BarlowTwinsTransform


class DVSDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset: str, timesteps: int = 10, data_dir: str = "data/", noise: str = None, severity=1, num_workers: int = 3, barlow_transf=None, mode="ann", **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dataset = dataset  # name of the dataset
        self.timesteps = timesteps
        self.noise = noise
        self.severity = severity
        self.num_workers = num_workers

        # create the directory if not exist
        os.makedirs(data_dir, exist_ok=True)

        # transform
        self.sensor_size, self.num_classes = self._get_dataset_info()
        self.train_transform = BarlowTwinsTransform(
            self.sensor_size, timesteps=timesteps, transforms_list=barlow_transf, concat_time_channels=mode=="ann")
        self.val_transform = BarlowTwinsTransform(self.sensor_size, timesteps=timesteps, transforms_list=barlow_transf)

    def _get_dataset_info(self):
        if self.dataset == "n-mnist":
            return tonic.datasets.NMNIST.sensor_size, len(tonic.datasets.NMNIST.classes)
        elif self.dataset == "cifar10-dvs":
            return CIFAR10DVS.sensor_size, 10
        elif self.dataset == "dvsgesture":
            return tonic.datasets.DVSGesture.sensor_size, len(tonic.datasets.DVSGesture.classes)
        elif self.dataset == "n-caltech101":
            return None, 101 # variable sensor_size for NCaltech
        elif self.dataset == "asl-dvs":
            return tonic.datasets.ASLDVS.sensor_size, len(tonic.datasets.ASLDVS.classes)
        elif self.dataset == 'ncars':
            return NCARS.sensor_size, len(NCARS.classes)

    def prepare_data(self) -> None:
        # downloads the dataset if it does not exist
        # NOTE: since we use the library named "Tonic", all the download process is handled, we just have to make an instanciation
        if self.dataset == "n-mnist":
            tonic.datasets.NMNIST(save_to=self.data_dir)
        elif self.dataset == "cifar10-dvs":
            CIFAR10DVS(save_to=self.data_dir)
        elif self.dataset == "dvsgesture":
            tonic.datasets.DVSGesture(save_to=self.data_dir)
        elif self.dataset == "n-caltech101":
            tonic.datasets.NCALTECH101(save_to=self.data_dir)
        elif self.dataset == "asl-dvs":
            tonic.datasets.ASLDVS(save_to=self.data_dir)
        elif self.dataset == 'ncars':
            NCARS(save_to=self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset == "n-mnist":
            self.train_set = tonic.datasets.NMNIST(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=True)
            self.val_set = tonic.datasets.NMNIST(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=False)

        elif self.dataset == "cifar10-dvs":
            dataset_train = CIFAR10DVS(save_to=self.data_dir, transform=self.train_transform, target_transform=None)
            dataset_val = CIFAR10DVS(save_to=self.data_dir, transform=self.train_transform, target_transform=None)
            full_len = len(dataset_train)
            train_len = int(0.80 * full_len)
            val_len = full_len - train_len
            print(full_len, train_len, val_len)
            self.train_set, _ = random_split(dataset_train, lengths=[train_len, val_len])
            _, self.val_set = random_split(dataset_val, lengths=[train_len, val_len])

        elif self.dataset == "dvsgesture":
            self.train_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=True)
            self.val_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=False)

        elif self.dataset == "n-caltech101":
            tonic.datasets.NCALTECH101(save_to=self.data_dir)

        elif self.dataset == "asl-dvs":
            dataset = tonic.datasets.ASLDVS(save_to=self.data_dir, transform=self.train_transform)
            full_length = len(dataset)
            print(full_length, 0.8 * full_length)
            self.train_set, _ = random_split(dataset, [0.8 * full_length, full_length - (0.8 * full_length)])
            dataset = tonic.datasets.ASLDVS(save_to=self.data_dir, transform=self.train_transform)
            _, self.val_set = random_split(dataset, [0.8 * full_length, full_length - (0.8 * full_length)])
        elif self.dataset == 'ncars':
            self.train_set = NCARS(self.data_dir, train=True, transform=self.train_transform)
            self.val_set = NCARS(self.data_dir, train=False, transform=self.train_transform)
            print(len(self.train_set), len(self.val_set))

        print(len(self.train_set), len(self.val_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
