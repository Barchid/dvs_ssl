from dataclasses import dataclass
from typing import Tuple
from cv2 import transform
from einops import rearrange
from sqlalchemy import Float
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.transforms import functional
from tonic import transforms as TF
import numpy as np
from project.utils.dvs_noises import BackgroundActivityNoise


class BarlowTwinsTransform:
    def __init__(self, sensor_size: int, timesteps: int = 10, transforms_list=[]):
        trans_a = []
        trans_b = []

        representation = get_frame_representation(sensor_size, timesteps)

        # BEFORE TENSOR TRANSFORMATION
        if 'flip' in transforms_list:
            trans_a.append(TF.RandomFlipLR(sensor_size))
            trans_b.append(TF.RandomFlipLR(sensor_size))

        if 'background_activity' in transforms_list:
            trans_a.append(transforms.RandomApply([
                BackgroundActivityNoise(sensor_size, severity=4)
            ], p=0.5))
            trans_b.append(transforms.RandomApply([
                BackgroundActivityNoise(sensor_size, severity=4)
            ], p=0.5))

        if 'hot_pixels' in transforms_list:
            pass

        if 'reverse' in transforms_list:
            trans_a.append(TF.RandomTimeReversal(p=0.2))  # only for transformation A (not B)
            trans_b.append(TF.RandomTimeReversal(p=0.2))  # only for transformation A (not B)

        if 'flip_polarity' in transforms_list:
            trans_a.append(TF.RandomFlipPolarity(p=0.5))
            trans_b.append(TF.RandomFlipPolarity(p=0.5))

        if 'time_jitter' in transforms_list:
            trans_a.append(TF.TimeJitter(clip_negative=True))
            trans_b.append(TF.TimeJitter(clip_negative=True))

        # TENSOR TRANSFORMATION
        trans_a.append(representation)
        trans_b.append(representation)

        if 'crop' in transforms_list:
            trans_a.append(transforms.RandomResizedCrop((sensor_size[0], sensor_size[1]), interpolation='nearest'))
            trans_b.append(transforms.RandomResizedCrop((sensor_size[0], sensor_size[1]), interpolation='nearest'))

        # AFTER TENSOR TRANSFORMATION
        if 'static_rotation' in transforms_list:
            trans_a.append(transforms.RandomRotation(10))  # Random rotation of [-10, 10] degrees)
            trans_b.append(transforms.RandomRotation(10))

        if 'static_translation' in transforms_list:
            trans_a.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))  # translation in Y and X axes
            trans_b.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))  # translation in Y and X axes

        if 'dynamic_rotation' in transforms_list:
            trans_a.append(DynamicRotation())
            trans_b.append(DynamicRotation())

        if 'dynamic_translation' in transforms_list:
            trans_a.append(DynamicTranslation())
            trans_b.append(DynamicTranslation())

        # finish by concatenating polarity and timesteps
        trans_a.append(
            transforms.Lambda(lambda x: rearrange(x, 'frames polarity height width -> (frames polarity) height width'))
        )
        trans_b.append(
            transforms.Lambda(lambda x: rearrange(x, 'frames polarity height width -> (frames polarity) height width'))
        )

        self.transform_a = transforms.Compose(trans_a)
        self.transfrom_b = transforms.Compose(trans_b)
        self.transform = transforms.Compose([
            representation,
            transforms.Lambda(lambda x: rearrange(x, 'frames polarity height width -> (frames polarity) height width'))
        ])

    def __call__(self, X):
        Y_a = self.transform_a(X)
        Y_b = self.transform_b(X)
        X = self.transform(X)
        return X, Y_a, Y_b


@dataclass(frozen=True)
class DynamicRotation:
    degrees: Tuple[Float] = (-10, 10)

    def __call__(self, frames: torch.Tensor):  # shape (..., H, W)
        timesteps = frames.shape[0]
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        step_angle = angle / (timesteps - 1)

        current_angle = 0.
        result = torch.zeros_like(frames)
        for t in range(timesteps):
            result[t] = functional.rotate(frames[t], current_angle)
            current_angle += step_angle

        return result[t]


@dataclass(frozen=True)
class DynamicTranslation:
    translate: Tuple[Float] = (0.1, 0.1)

    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        timesteps, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]

        # compute max translation
        max_dx = float(self.translate[0] * H)
        max_dy = float(self.translate[1] * W)
        max_tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        max_ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))

        # step translation
        step_tx = max_tx // (timesteps - 1)
        current_tx = 0

        step_ty = max_ty // (timesteps - 1)
        current_ty = 0

        result = torch.zeros_like(frames)
        for t in timesteps:
            translations = (current_tx, current_ty)
            result[t] = functional.affine(frames[t], 0., translate=translations, scale=0., shear=0.)
            current_tx += step_tx
            current_ty += step_ty

        return result[t]


def get_frame_representation(sensor_size, timesteps):
    return transforms.Compose([
        TF.ToFrame(sensor_size, n_time_bins=timesteps),
        transforms.Lambda(lambda x: (x > 0).astype(np.float32)),
        transforms.Lambda(lambda x: torch.from_numpy(x))
    ])
