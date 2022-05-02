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
from snntorch.spikegen import delta

from project.utils.transform_dvs import BackgroundActivityNoise, RandomFlipLR, ToFrame


class BarlowTwinsTransform:
    def __init__(self, sensor_size = None, timesteps: int = 10, transforms_list=[], concat_time_channels = True):
        trans_a = []
        trans_b = []

        representation = get_frame_representation(sensor_size, timesteps)

        # BEFORE TENSOR TRANSFORMATION
        if 'flip' in transforms_list:
            trans_a.append(RandomFlipLR(sensor_size=sensor_size))
            trans_b.append(RandomFlipLR(sensor_size=sensor_size))

        if 'background_activity' in transforms_list:
            trans_a.append(transforms.RandomApply([
                BackgroundActivityNoise(severity=4, sensor_size=sensor_size)
            ], p=0.5))
            trans_b.append(transforms.RandomApply([
                BackgroundActivityNoise(severity=4, sensor_size=sensor_size)
            ], p=0.5))

        if 'hot_pixels' in transforms_list:
            pass

        if 'reverse' in transforms_list:
            trans_a.append(TF.RandomTimeReversal(p=0.2))  # only for transformation A (not B)
            trans_b.append(TF.RandomTimeReversal(p=0.2))  # only for transformation A (not B)

        if 'flip_polarity' in transforms_list:
            trans_a.append(TF.RandomFlipPolarity(p=0.35))
            trans_b.append(TF.RandomFlipPolarity(p=0.35))

        if 'time_jitter' in transforms_list:
            trans_a.append(TF.TimeJitter(clip_negative=True))
            trans_b.append(TF.TimeJitter(clip_negative=True))

        # TENSOR TRANSFORMATION
        trans_a.append(representation)
        trans_b.append(representation)

        # if 'crop' in transforms_list:
        # trans_a.append(transforms.RandomResizedCrop((224, 224), interpolation=transforms.InterpolationMode.NEAREST))
        trans_a.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)) # debug
        # trans_b.append(transforms.RandomResizedCrop((224, 224), interpolation=transforms.InterpolationMode.NEAREST))
        trans_b.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)) # debug

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

        if 'moving_occlusion' in transforms_list:
            trans_a.append(MovingOcclusion())
            trans_b.append(MovingOcclusion())

        if 'cutout' in transforms_list:
            trans_a.append(Cutout())
            trans_b.append(Cutout())

        # finish by concatenating polarity and timesteps
        if concat_time_channels:
            trans_a.append(
                transforms.Lambda(lambda x: rearrange(x, 'frames polarity height width -> (frames polarity) height width'))
            )
            trans_b.append(
                transforms.Lambda(lambda x: rearrange(x, 'frames polarity height width -> (frames polarity) height width'))
            )
            
            self.transform = transforms.Compose([
                representation,
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Lambda(lambda x: rearrange(x, 'frames polarity height width -> (frames polarity) height width'))
            ])
        else:
            self.transform = transforms.Compose([
                representation,
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
            ])

        self.transform_a = transforms.Compose(trans_a)
        self.transform_b = transforms.Compose(trans_b)

    def __call__(self, X):
        Y_a = self.transform_a(X)
        Y_b = self.transform_b(X)
        X = self.transform(X)
        return X, Y_a, Y_b


@dataclass(frozen=True)
class DynamicRotation:
    degrees: Tuple[float] = (-10, 10)

    def __call__(self, frames: torch.Tensor):  # shape (..., H, W)
        timesteps = frames.shape[0]
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        step_angle = angle / (timesteps - 1)

        current_angle = 0.
        result = torch.zeros_like(frames)
        for t in range(timesteps):
            result[t] = functional.rotate(frames[t], current_angle)
            current_angle += step_angle

        return result


@dataclass(frozen=True)
class DynamicTranslation:
    translate: Tuple[float] = (0.1, 0.1)

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
        for t in range(timesteps):
            translations = (current_tx, current_ty)
            result[t] = functional.affine(frames[t], 0., translate=translations, scale=0., shear=0., fill=0)
            current_tx += step_tx
            current_ty += step_ty

        return result[t]


def get_frame_representation(sensor_size, timesteps):
    return transforms.Compose([
        ToFrame(sensor_size=sensor_size, n_time_bins=timesteps),
        transforms.Lambda(lambda x: (x > 0).astype(np.float32)),
        transforms.Lambda(lambda x: torch.from_numpy(x))
    ])


@dataclass(frozen=True)
class Cutout:
    size: Tuple[float] = (0.1, 0.2)
    nb_holes: int = 4

    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        timesteps, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]

        mask = torch.ones_like(frames)
        for i in range(self.nb_holes):
            # compute size of the
            size = random.uniform(self.size[0], self.size[1])
            size_h = int(H * size)
            size_w = int(W * size)
            x_min, y_min = random.randint(0, W - size_w), random.randint(0, H - size_h)
            x_max, y_max = x_min + size_w, y_min + size_h
            mask[:, :, y_min:(y_max + 1), x_min:(x_max + 1)] = 0.

        # drop events where the
        frames[mask == 0] = 0.

        return frames


@dataclass(frozen=True)
class MovingOcclusion:
    size: Tuple[float] = (0.1, 0.2)
    nb_holes: int = 4
    translate: Tuple[float] = (0.3, 0.3)

    def _hole_translation(self, mask: torch.Tensor, H, W, timesteps):
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

        translated = torch.zeros((timesteps, H, W))  # shape=(T,H,W)
        for t in range(timesteps):
            translations = (current_tx, current_ty)
            translated[t] = functional.affine(mask, 0., translate=translations, scale=0., shear=0., fill=0)
            current_tx += step_tx
            current_ty += step_ty

        translated = delta(translated, padding=True, off_spike=True)  # composed of 1's and -1's for the polarities

        result = torch.zeros((timesteps, 2, H, W))  # shape=(T,C,H,W)
        for t in range(timesteps):
            result[t, 0, :, :] = (translated[t] == 1.).type(torch.float32)  # positive events
            result[t, 1, :, :] = (translated[t] == -1.).type(torch.float32)  # negative events

        return result

    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        timesteps, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]

        # each hole creates a mask and is translated randomly, then it is added to the frames result
        for i in range(self.nb_holes):
            # create hole
            mask = torch.ones((H, W))  # shape=(H,W)
            size = random.uniform(self.size[0], self.size[1])
            size_h = int(H * size)
            size_w = int(W * size)
            x_min, y_min = random.randint(0, W - size_w), random.randint(0, H - size_h)
            x_max, y_max = x_min + size_w, y_min + size_h
            mask[y_min:(y_max + 1), x_min:(x_max + 1)] = 0.

            is_on = random.choice([True, False])  # choice if the hole is ON polarity or OFF polarity
            if is_on:
                mask = (~(mask > 0)).type(torch.float32)

            # random translation
            hole = self._hole_translation(mask, H, W, timesteps)  # hole.shape=(T, C, H, W)

            # drop events where the mask
            frames = torch.logical_or(frames, hole, out=torch.empty_like(frames))

        return frames
