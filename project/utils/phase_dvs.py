from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tonic
import numpy as np
import cv2
from dataclasses import dataclass


def to_timesurface_custom(
    events, sensor_size, tau=5e3, decay="lin"
):

    radius_x = 0
    radius_y = 0
    surface_dimensions = sensor_size

    assert "x" and "y" and "t" and "p" in events.dtype.names

    timestamp_memory = np.zeros(
        (sensor_size[2], sensor_size[1] + radius_y * 2, sensor_size[0] + radius_x * 2)
    )
    timestamp_memory -= tau * 3 + 1
    result = np.zeros_like(timestamp_memory)
    for index, event in enumerate(events):
        x = int(event["x"])
        y = int(event["y"])
        timestamp_memory[int(event["p"]), y + radius_y, x + radius_x] = event["t"]
        timestamp_context = timestamp_memory - event["t"]

        if decay == "lin":
            timesurface = timestamp_context / (3 * tau) + 1
            timesurface[timesurface < 0] = 0
        elif decay == "exp":
            timesurface = np.exp(timestamp_context / tau)
        result += timesurface

    return result / len(events)


@dataclass(frozen=True)
class ToTimeSurfaceCustom:
    sensor_size: Tuple[int, int, int]
    # tau: float
    # decay: str

    def __call__(self, events):
        return to_timesurface_custom(events.copy(), self.sensor_size)


def to_bit_encoding(events: np.ndarray, sensor_size, timesteps: int = 1):
    event_frames = tonic.transforms.functional.to_frame_numpy(events, sensor_size, n_time_bins=timesteps * 8)
    event_frames = (event_frames > 0).astype(np.uint8)  # binary event frames
    res = np.zeros((timesteps, *event_frames.shape[1:]), dtype=np.float32)
    for i in range(timesteps):
        e_f = event_frames[i*8:(i+1)*8]
        bit_frame = np.packbits(e_f, axis=0).astype(np.float32) / 255.
        res[i] = bit_frame[0]
    return res


def to_weighted_frames(events: np.ndarray, sensor_size, timesteps: int, blur_type=None):
    event_frames = tonic.transforms.functional.to_frame_numpy(events, sensor_size, n_time_bins=timesteps)
    event_frames = (event_frames > 0).astype(np.float32)  # binary event frames

    weight = 1. / timesteps
    event_frames = event_frames * weight
    frames = event_frames.sum(0)

    if blur_type is not None:
        frames = to_blur(frames, blur_type)

    return frames


@dataclass(frozen=True)
class ToBitEncoding:
    sensor_size: Tuple[int, int, int]
    timesteps: int

    def __call__(self, events):
        return to_bit_encoding(events.copy(), self.sensor_size, timesteps=self.timesteps)


@dataclass(frozen=True)
class ToWeightedFrames:
    sensor_size: Tuple[int, int, int]
    timesteps: int
    blur_type: str

    def __call__(self, events):
        return to_weighted_frames(events.copy(), self.sensor_size, self.timesteps, self.blur_type)


def to_blur(event_frames: np.ndarray, blur_type: str = 'averaging'):
    if blur_type == 'averaging':
        return cv2.blur(event_frames, (5, 5))

    elif blur_type == 'median':
        return cv2.medianBlur(event_frames, 5)

    elif blur_type == 'gaussian':
        return cv2.GaussianBlur(event_frames, (5, 5), 0)

    elif blur_type == 'bilateral':
        return cv2.bilateralFilter(event_frames, 9, 75, 75)

    else:
        NotImplementedError('Must implement other blur strategies before using them.')
