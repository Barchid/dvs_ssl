from dataclasses import dataclass
from typing import Tuple
from cv2 import transform
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.transforms import functional
from tonic import transforms as TF
import numpy as np
import random
import tonic
from project.datamodules.cifar10dvs import CIFAR10DVS
from project.datamodules.dvs_lips import DVSLip
from project.datamodules.ncaltech101 import NCALTECH101
from project.datamodules.ncars import NCARS
from project.utils.barlow_transforms import (
    Cutout,
    DynamicRotation,
    DynamicTranslation,
    TransRot,
)
from project.utils.dvs_noises import EventDrop, EventDrop2, EventDrop3
from project.utils.transform_dvs import (
    BackgroundActivityNoise,
    ConcatTimeChannels,
    CutMixEvents,
    CutPasteEvent,
    MovingOcclusion,
    RandomFlipLR,
    RandomFlipPolarity,
    RandomTimeReversal,
    ToFrame,
    get_frame_representation,
)


class SupervisedTransform:
    def __init__(
        self,
        sensor_size=None,
        timesteps: int = 10,
        transforms_list=[],
        concat_time_channels=True,
        dataset=None,
        data_dir=None,
    ):
        self.trans_bef = {}
        self.trans_aft = {}
        self.transforms_list = transforms_list
        self.representation = get_frame_representation(sensor_size, timesteps)

        # BEFORE TENSOR TRANSFORMATION
        self.flip = None
        if "flip" in transforms_list:
            self.transforms_list.remove("flip")
            self.flip = RandomFlipLR(sensor_size=sensor_size)
            # trans_a.append(RandomFlipLR(sensor_size=sensor_size))
            # trans_b.append(RandomFlipLR(sensor_size=sensor_size))

        if "background_activity" in transforms_list:
            self.trans_bef["background_activity"] = BackgroundActivityNoise(
                severity=5, sensor_size=sensor_size
            )

        if "hot_pixels" in transforms_list:
            pass

        if "reverse" in transforms_list:
            self.trans_bef["reverse"] = RandomTimeReversal(p=1.0)

        if "flip_polarity" in transforms_list:
            self.trans_bef["flip_polarity"] = RandomFlipPolarity(p=1.0)

        if "cutpaste" in transforms_list:
            self.trans_bef["cutpaste"] = CutPasteEvent(sensor_size=sensor_size)

        if "event_drop" in transforms_list:
            self.trans_bef["event_drop"] = EventDrop(sensor_size=sensor_size)

        if "event_drop_2" in transforms_list:
            self.trans_bef["event_drop_2"] = EventDrop2(sensor_size=sensor_size)

        # if 'crop' in transforms_list:
        if "crop" in transforms_list:
            self.transforms_list.remove("crop")
            self.crop = transforms.RandomResizedCrop(
                (128, 128), interpolation=transforms.InterpolationMode.NEAREST
            )
        else:
            self.crop = transforms.Resize(
                (128, 128), interpolation=transforms.InterpolationMode.NEAREST
            )

        # AFTER TENSOR TRANSFORMATION
        if "static_rotation" in transforms_list:
            # Random rotation of [-20, 20] degrees)
            self.trans_aft["static_rotation"] = transforms.RandomRotation(75)

        if "static_translation" in transforms_list:
            self.trans_aft["static_translation"] = transforms.RandomAffine(
                0, translate=(0.2, 0.2)
            )

        if "dynamic_rotation" in transforms_list:
            self.trans_aft["dynamic_rotation"] = DynamicRotation()

        if "dynamic_translation" in transforms_list:
            self.trans_aft["dynamic_translation"] = DynamicTranslation()

        if "transrot" in transforms_list:
            self.trans_aft["transrot"] = TransRot()

        # if "moving_occlusion" in transforms_list:
        #     trans_a.append(transforms.RandomApply([MovingOcclusion()], p=0.5))
        #     trans_b.append(transforms.RandomApply([MovingOcclusion()], p=0.5))

        if "cutout" in transforms_list:
            self.trans_aft["cutout"] = Cutout()

        # finish by concatenating polarity and timesteps
        if concat_time_channels:
            self.concat = ConcatTimeChannels()

            self.transform = transforms.Compose(
                [
                    self.representation,
                    transforms.Resize(
                        (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    ConcatTimeChannels(),
                ]
            )
        else:
            self.concat = None
            self.transform = transforms.Compose(
                [
                    self.representation,
                    transforms.Resize(
                        (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                    ),
                ]
            )

    def __call__(self, X):
        k = 2 if len(self.transforms_list) >= 2 else 1
        smpls = random.sample(self.transforms_list, k=k)
        if self.flip is not None:
            final_transforms = [self.flip]
        else:
            final_transforms = []

        for smpl in smpls:
            if smpl in self.trans_bef:
                final_transforms.append(self.trans_bef[smpl])

        final_transforms.append(self.representation)

        final_transforms.append(self.crop)

        for smpl in smpls:
            if smpl in self.trans_aft:
                final_transforms.append(self.trans_aft[smpl])

        if self.concat is not None:
            final_transforms.append(self.concat)

        final_transforms = transforms.Compose(final_transforms)
        Y_a = final_transforms(X)
        X = self.transform(X)
        return X, Y_a, X
