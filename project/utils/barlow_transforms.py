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


class BarlowTwinsTransform:
    def __init__(
        self,
        sensor_size=None,
        timesteps: int = 10,
        transforms_list=[],
        concat_time_channels=True,
        dataset=None,
        data_dir=None,
    ):
        trans_a = []
        trans_b = []

        representation = get_frame_representation(sensor_size, timesteps, dataset=dataset)

        # BEFORE TENSOR TRANSFORMATION
        if "flip" in transforms_list:
            trans_a.append(RandomFlipLR(sensor_size=sensor_size))
            trans_b.append(RandomFlipLR(sensor_size=sensor_size))

        if "background_activity" in transforms_list:
            trans_a.append(
                transforms.RandomApply(
                    [BackgroundActivityNoise(severity=5, sensor_size=sensor_size)],
                    p=0.5,
                )
            )
            trans_b.append(
                transforms.RandomApply(
                    [BackgroundActivityNoise(severity=5, sensor_size=sensor_size)],
                    p=0.5,
                )
            )

        if "hot_pixels" in transforms_list:
            pass

        if "reverse" in transforms_list:
            trans_a.append(
                RandomTimeReversal(p=0.2)
            )  # only for transformation A (not B)
            trans_b.append(
                RandomTimeReversal(p=0.2)
            )  # only for transformation A (not B)

        if "flip_polarity" in transforms_list:
            trans_a.append(RandomFlipPolarity(p=0.2))
            trans_b.append(RandomFlipPolarity(p=0.2))

        if "time_jitter" in transforms_list:
            trans_a.append(TF.TimeJitter(clip_negative=True))
            trans_b.append(TF.TimeJitter(clip_negative=True))

        if "cutmix" in transforms_list:
            # NOTE: since we use the library named "Tonic", all the download process is handled, we just have to make an instanciation
            if dataset == "n-mnist":
                dataset = tonic.datasets.NMNIST(save_to=data_dir)
            elif dataset == "cifar10-dvs":
                dataset = CIFAR10DVS(save_to=data_dir)
            elif dataset == "dvsgesture":
                dataset = tonic.datasets.DVSGesture(save_to=data_dir)
            elif dataset == "n-caltech101":
                dataset = NCALTECH101(save_to=data_dir)
            elif dataset == "asl-dvs":
                tonic.datasets.ASLDVS(save_to=data_dir)
            elif dataset == "ncars":
                dataset = NCARS(save_to=data_dir, download=True)
            elif dataset == "dvs_lips":
                dataset = DVSLip(save_to=data_dir)

            trans_a.append(
                transforms.RandomApply(
                    [CutMixEvents(dataset, sensor_size=sensor_size)], p=0.5
                )
            )
            trans_b.append(
                transforms.RandomApply(
                    [CutMixEvents(dataset, sensor_size=sensor_size)], p=0.5
                )
            )

        if "eventmix" in transforms_list:
            # NOTE: since we use the library named "Tonic", all the download process is handled, we just have to make an instanciation
            if dataset == "n-mnist":
                dataset = tonic.datasets.NMNIST(save_to=data_dir)
            elif dataset == "cifar10-dvs":
                dataset = CIFAR10DVS(save_to=data_dir)
            elif dataset == "dvsgesture":
                dataset = tonic.datasets.DVSGesture(save_to=data_dir)
            elif dataset == "n-caltech101":
                dataset = NCALTECH101(save_to=data_dir)
            elif dataset == "asl-dvs":
                tonic.datasets.ASLDVS(save_to=data_dir)
            elif dataset == "ncars":
                dataset = NCARS(save_to=data_dir, download=True)
            elif dataset == "dvs_lips":
                dataset = DVSLip(save_to=data_dir)

            trans_a.append(
                transforms.RandomApply(
                    [CutMixEvents(dataset, sensor_size=sensor_size)], p=0.5
                )
            )
            trans_b.append(
                transforms.RandomApply(
                    [CutMixEvents(dataset, sensor_size=sensor_size)], p=0.5
                )
            )

        if "cutpaste" in transforms_list:
            trans_a.append(
                transforms.RandomApply([CutPasteEvent(sensor_size=sensor_size)], p=0.5)
            )
            trans_b.append(
                transforms.RandomApply([CutPasteEvent(sensor_size=sensor_size)], p=0.5)
            )

        if "event_drop" in transforms_list:
            trans_a.append(EventDrop(sensor_size=sensor_size))
            trans_b.append(EventDrop(sensor_size=sensor_size))

        if "event_drop_2" in transforms_list:
            trans_a.append(EventDrop2(sensor_size=sensor_size))
            trans_b.append(EventDrop2(sensor_size=sensor_size))

        if "event_drop_3" in transforms_list:
            trans_a.append(EventDrop3(sensor_size=sensor_size))
            trans_b.append(EventDrop3(sensor_size=sensor_size))

        # TENSOR TRANSFORMATION
        trans_a.append(representation)
        trans_b.append(representation)

        if dataset == "dvs_lips":
            out_size = (128, 128)
        else:
            out_size = (128, 128)
            
        # if 'crop' in transforms_list:
        if "crop" in transforms_list:
            trans_a.append(
                transforms.RandomResizedCrop(
                    out_size, interpolation=transforms.InterpolationMode.NEAREST
                )
            )
            trans_b.append(
                transforms.RandomResizedCrop(
                    out_size, interpolation=transforms.InterpolationMode.NEAREST
                )
            )
        else:
            trans_a.append(
                transforms.Resize(
                    out_size, interpolation=transforms.InterpolationMode.NEAREST
                )
            )
            trans_b.append(
                transforms.Resize(
                    out_size, interpolation=transforms.InterpolationMode.NEAREST
                )
            )  # debug

        # AFTER TENSOR TRANSFORMATION
        if "static_rotation" in transforms_list:
            # Random rotation of [-20, 20] degrees)
            trans_a.append(
                transforms.RandomApply([transforms.RandomRotation(75)], p=0.5)
            )
            # Random rotation of [-20, 20] degrees)
            trans_b.append(
                transforms.RandomApply([transforms.RandomRotation(75)], p=0.5)
            )

        if "static_translation" in transforms_list:
            trans_a.append(
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5
                )
            )  # translation in Y and X axes
            trans_b.append(
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5
                )
            )  # translation in Y and X axes

        if "dynamic_rotation" in transforms_list:
            trans_a.append(transforms.RandomApply([DynamicRotation()], p=0.5))
            trans_b.append(transforms.RandomApply([DynamicRotation()], p=0.5))

        if "dynamic_translation" in transforms_list:
            trans_a.append(transforms.RandomApply([DynamicTranslation()], p=0.5))
            trans_b.append(transforms.RandomApply([DynamicTranslation()], p=0.5))

        if "transrot" in transforms_list:
            trans_a.append(TransRot())
            trans_b.append(TransRot())

        if "moving_occlusion" in transforms_list:
            trans_a.append(transforms.RandomApply([MovingOcclusion()], p=0.5))
            trans_b.append(transforms.RandomApply([MovingOcclusion()], p=0.5))

        if "cutout" in transforms_list:
            trans_a.append(transforms.RandomApply([Cutout()], p=0.3))
            trans_b.append(transforms.RandomApply([Cutout()], p=0.3))

        # finish by concatenating polarity and timesteps
        if concat_time_channels:
            trans_a.append(ConcatTimeChannels())
            trans_b.append(ConcatTimeChannels())

            self.transform = transforms.Compose(
                [
                    representation,
                    transforms.Resize(
                        out_size, interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    ConcatTimeChannels(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    representation,
                    transforms.Resize(
                        out_size, interpolation=transforms.InterpolationMode.NEAREST
                    ),
                ]
            )

        self.transform_a = transforms.Compose(trans_a)
        self.transform_b = transforms.Compose(trans_b)

    def __call__(self, X):
        Y_a = self.transform_a(X)
        Y_b = self.transform_b(X)
        X = self.transform(X)
        return X, Y_a, Y_b


@dataclass(frozen=True)
class DynamicRotation:
    degrees: Tuple[float] = (-75, 75)

    def __call__(self, frames: torch.Tensor):  # shape (..., H, W)
        timesteps = frames.shape[0]
        angle = float(
            torch.empty(1)
            .uniform_(float(self.degrees[0]), float(self.degrees[1]))
            .item()
        )
        step_angle = angle / (timesteps - 1)

        current_angle = 0.0
        result = torch.zeros_like(frames)
        for t in range(timesteps):
            result[t] = functional.rotate(frames[t], current_angle)
            current_angle += step_angle

        return result


@dataclass(frozen=True)
class DynamicTranslation:
    translate: Tuple[float] = (0.3, 0.3)

    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        timesteps, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]

        # compute max translation
        max_dx = float(self.translate[0] * H)
        max_dy = float(self.translate[1] * W)
        max_tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        max_ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        # step translation
        step_tx = max_tx / (timesteps - 1)
        current_tx = 0

        step_ty = max_ty / (timesteps - 1)
        current_ty = 0

        result = torch.zeros_like(frames)
        for t in range(timesteps):
            translations = (round(current_tx), round(current_ty))
            result[t] = functional.affine(
                frames[t], 0.0, translate=translations, scale=1.0, shear=0.0, fill=0
            )
            current_tx += step_tx
            current_ty += step_ty

        return result


@dataclass(frozen=True)
class TransRot:
    dyn_tran = DynamicTranslation()
    dyn_rot = DynamicRotation()
    stat_tran = transforms.RandomAffine(0, translate=(0.2, 0.2))
    stat_rot = transforms.RandomRotation(75)

    def __call__(self, frames: torch.Tensor):
        choice = np.random.randint(0, 5)
        if choice == 0:
            return frames
        if choice == 1:
            return self.dyn_tran(frames)
        if choice == 2:
            return self.dyn_rot(frames)
        if choice == 3:
            return self.stat_tran(frames)
        if choice == 4:
            return self.stat_rot(frames)


@dataclass(frozen=True)
class Cutout:
    size: Tuple[float] = (0.3, 0.6)
    nb_holes: int = 3

    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        timesteps, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]

        mask = torch.ones_like(frames)
        n_holes = random.randint(1, self.nb_holes)
        for i in range(n_holes):
            # compute size of the
            size = random.uniform(self.size[0], self.size[1])
            size_h = int(H * size)
            size_w = int(W * size)
            x_min, y_min = random.randint(0, W - size_w), random.randint(0, H - size_h)
            x_max, y_max = x_min + size_w, y_min + size_h
            mask[:, :, y_min : (y_max + 1), x_min : (x_max + 1)] = 0.0

        # drop events where the
        frames[mask == 0] = 0.0

        return frames
