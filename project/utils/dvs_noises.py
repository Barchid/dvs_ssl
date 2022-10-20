from dataclasses import dataclass
import os
from typing import Optional, Tuple
import numpy as np
import tonic
from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
from tonic import functional as TF
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import cv2
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from project.utils.drop_event import (
    drop_by_area_numpy,
    drop_by_time_numpy,
    drop_event_numpy,
)
from project.utils.transform_dvs import CutPasteEvent, MovingOcclusion


@dataclass(frozen=True)
class RandomTimeReversal:
    """Temporal flip is defined as:

        .. math::
           t_i' = max(t) - t_i

           p_i' = -1 * p_i

    Parameters:
        p (float): probability of performing the flip
    """

    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "t" and "p" in events.dtype.names
        if np.random.rand() < self.p:
            events["t"] = np.max(events["t"]) - events["t"]
            if events["p"].dtype == np.int64:
                events["p"] *= -1
            else:
                events["p"] = np.invert(events["p"])
        return events


@dataclass(frozen=True)
class EventDrop:
    """Applies EventDrop transformation from the paper "EventDrop: Data Augmentation for Event-based Learning".
        Applies one of the 4 drops of event strategies between:
            1. Identity (do nothing)
            2. Drop events by time
            3. Drop events by area
            4. Drop events randomly

        For each strategy, the ratio of dropped events are determined in the paper.

    Args:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]

    Example:
        >>> transform = tonic.transforms.EventDrop(sensor_size=(128,128,2))
    """

    sensor_size: Tuple[int, int, int]

    def __call__(self, events):
        choice = np.random.randint(0, 4)
        if choice == 0:
            return events
        if choice == 1:
            duration_ratio = np.random.randint(1, 10) / 10.0
            return drop_by_time_numpy(events, duration_ratio)
        if choice == 2:
            area_ratio = np.random.randint(1, 6) / 20.0
            return drop_by_area_numpy(events, self.sensor_size, area_ratio)
        if choice == 3:
            ratio = np.random.randint(1, 10) / 10.0
            return drop_event_numpy(events, ratio)


@dataclass(frozen=True)
class EventDrop2:
    """Applies EventDrop transformation from the paper "EventDrop: Data Augmentation for Event-based Learning".
        Applies one of the 4 drops of event strategies between:
            1. Identity (do nothing)
            2. Drop events by time
            3. Drop events by area
            4. Drop events randomly
            5. EventCutPaste

        For each strategy, the ratio of dropped events are determined in the paper.

    Args:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]

    Example:
        >>> transform = tonic.transforms.EventDrop(sensor_size=(128,128,2))
    """

    sensor_size: Tuple[int, int, int]
    cutpaste = CutPasteEvent()

    def __call__(self, events):
        choice = np.random.randint(0, 5)
        if choice == 0:
            return events
        if choice == 1:
            duration_ratio = np.random.randint(1, 10) / 10.0
            return drop_by_time_numpy(events, duration_ratio)
        if choice == 2:
            area_ratio = np.random.randint(1, 6) / 20.0
            return drop_by_area_numpy(events, self.sensor_size, area_ratio)
        if choice == 3:
            ratio = np.random.randint(1, 10) / 10.0
            return drop_event_numpy(events, ratio)
        if choice == 4:
            return self.cutpaste(events)
        
        
@dataclass(frozen=True)
class EventDrop3:
    """Applies EventDrop transformation from the paper "EventDrop: Data Augmentation for Event-based Learning".
        Applies one of the 4 drops of event strategies between:
            1. Identity (do nothing)
            2. Drop events by time
            3. Drop events by area
            4. Drop events randomly
            5. EventCutPaste MovingOcc

        For each strategy, the ratio of dropped events are determined in the paper.

    Args:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]

    Example:
        >>> transform = tonic.transforms.EventDrop(sensor_size=(128,128,2))
    """

    sensor_size: Tuple[int, int, int]
    cutpaste = CutPasteEvent()
    movingocc = MovingOcclusion()

    def __call__(self, events):
        choice = np.random.randint(0, 6)
        if choice == 0:
            return events
        if choice == 1:
            duration_ratio = np.random.randint(1, 10) / 10.0
            return drop_by_time_numpy(events, duration_ratio)
        if choice == 2:
            area_ratio = np.random.randint(1, 6) / 20.0
            return drop_by_area_numpy(events, self.sensor_size, area_ratio)
        if choice == 3:
            ratio = np.random.randint(1, 10) / 10.0
            return drop_event_numpy(events, ratio)
        if choice == 4:
            return self.movingocc(events)
        if choice == 5:
            return self.cutpaste(events)


@dataclass
class CenteredOcclusion:
    severity: int
    sensor_size: Tuple[int, int, int]

    def __call__(self, events):
        c = [0.35, 0.45, 0.50, 0.60, 0.70][self.severity - 1]  # c is the sigma here
        mid = (self.sensor_size[0] // 2, self.sensor_size[1] // 2, self.sensor_size[2])

        occ_len_x = int(self.sensor_size[0] * c)
        occ_len_y = int(self.sensor_size[1] * c)

        # get coordinates of a centered crop
        coordinates = []
        for x in range(mid[0] - occ_len_x // 2, mid[0] + occ_len_x // 2):
            for y in range(mid[1] - occ_len_y // 2, mid[1] + occ_len_y // 2):
                coordinates.append((x, y))

        return tonic.transforms.functional.drop_pixel_numpy(
            events=events, coordinates=coordinates
        )


@dataclass(frozen=True)
class HotPixelActivty:
    sensor_size: Tuple[int, int, int]
    severity: int

    def __call__(self, events):
        c = [0.03, 0.06, 0.09, 0.17, 0.27][
            self.severity - 1
        ]  # percentage of events to add in noise
        n_noise_events = c * len(events)

        sensor_size = [4, 4]

        noise_events = np.zeros(n_noise_events, dtype=events.dtype)
        for channel in events.dtype.names:
            event_channel = events[channel]
            if channel == "x":
                low, high = 0, self.sensor_size[0]
            if channel == "y":
                low, high = 0, self.sensor_size[1]
            if channel == "p":
                low, high = 0, self.sensor_size[2]
            if channel == "t":
                low, high = events["t"].min(), events["t"].max()
            noise_events[channel] = np.random.uniform(
                low=low, high=high, size=n_noise_events
            )
        events = np.concatenate((events, noise_events))
        return events[np.argsort(events["t"])]


def hot_pixels(frames: np.array, severity: int):
    # frames dim : (T,C,H,W)
    # severity between 1 and 5
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]

    # height and with of dvs frames
    height, width = frames.shape[-2], frames.shape[-1]

    # create the mask of hot pixels (H,W) (1 where there will be a hot pixel and 0 where the pixels won't be broken)
    # from https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones
    N = frames.shape[-2] * frames.shape[-1]  # total size of mask
    K = int(c * N)  # certain proportion of broken pixels
    hot_pixels_mask = np.array([1.0] * K + [0.0] * (N - K))
    np.random.shuffle(hot_pixels_mask)
    hot_pixels_mask = np.reshape(hot_pixels_mask, (height, width))

    # apply hot pixels in frames
    result = []
    for i in range(frames.shape[0]):
        frame = frames[i]  # C,H,W or # H, W
        if frame.ndim == 3:
            frame[0][hot_pixels_mask == 1] = 1.0  # for positive channel
            frame[1][hot_pixels_mask == 1] = 1.0  # for negative channel
        else:
            frame[hot_pixels_mask == 1] = 1.0  # for positive channel
        result.append(frame)

    return np.array(result, dtype=np.float32)


def spatial_jitter(events, sensor_size, severity):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]  # c is the sigma here
    variance = c**2
    return TF.spatial_jitter_numpy(
        events,
        sensor_size,
        variance_x=variance,
        variance_y=variance,
        clip_outliers=True,
    )


def temporal_jitter(events, severity):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]  # c is the sigma here
    return TF.time_jitter_numpy(events, c, clip_negative=True, sort_timestamps=True)


def background_activity(frames: np.ndarray, severity: int):
    # c is the average rate of background activity noise here
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]

    noise_mask = np.random.poisson(lam=c, size=frames.shape)

    frames = np.clip(frames + noise_mask, 0, 1).astype(np.float32)
    return frames
