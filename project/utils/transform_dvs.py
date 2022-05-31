from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tonic.transforms import functional
from torchvision import transforms
import random


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
            # events = events[np.argsort(events["t"])]
            events = np.flip(events)
            # events["p"] *= -1
            events['p'] = np.logical_not(events['p'])  # apply to boolean (inverse)
        return events


@dataclass(frozen=True)
class RandomFlipPolarity:
    """Flips polarity of individual events with p.
    Changes polarities 1 to -1 and polarities [-1, 0] to 1

    Parameters:
        p (float): probability of flipping individual event polarities
    """

    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "p" in events.dtype.names
        # flips = np.ones(len(events))
        probs = np.random.rand(len(events))
        mask = probs < self.p
        events["p"][mask] = np.logical_not(events['p'][mask])
        return events


@dataclass(frozen=True)
class RandomFlipLR:
    """Flips events in x. Pixels map as:

        x' = width - x

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        p (float): probability of performing the flip
    """

    sensor_size: Tuple[int, int, int] = None
    p: float = 0.5

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size

        events = events.copy()
        assert "x" in events.dtype.names
        if np.random.rand() <= self.p:
            events["x"] = sensor_size[0] - 1 - events["x"]
        return events


@dataclass(frozen=True)
class ToFrame:
    """Accumulate events to frames by slicing along constant time (time_window),
    constant number of events (spike_count) or constant number of frames (n_time_bins / n_event_bins).
    You can set one of the first 4 parameters to choose the slicing method. Depending on which method you choose,
    overlap will assume different functionality, whether that might be temporal overlap, number of events
    or fraction of a bin. As a rule of thumb, here are some considerations if you are unsure which slicing
    method to choose:

    * If your recordings are of roughly the same length, a safe option is to set time_window. Bare in mind
      that the number of events can vary greatly from slice to slice, but will give you some consistency when
      training RNNs or other algorithms that have time steps.

    * If your recordings have roughly the same amount of activity / number of events and you are more interested
      in the spatial composition, then setting spike_count will give you frames that are visually more consistent.

    * The previous time_window and spike_count methods will likely result in a different amount of frames for each
      recording. If your training method benefits from consistent number of frames across a dataset (for easier
      batching for example), or you want a parameter that is easier to set than the exact window length or number
      of events per slice, consider fixing the number of frames by setting n_time_bins or n_event_bins. The two
      methods slightly differ with respect to how the slices are distributed across the recording. You can define
      an overlap between 0 and 1 to provide some robustness.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        time_window (float): time window length for one frame. Use the same time unit as timestamps in the event recordings.
                             Good if you want temporal consistency in your training, bad if you need some visual consistency
                             for every frame if the recording's activity is not consistent.
        spike_count (int): number of events per frame. Good for training CNNs which do not care about temporal consistency.
        n_time_bins (int): fixed number of frames, sliced along time axis. Good for generating a pre-determined number of
                           frames which might help with batching.
        n_event_bins (int): fixed number of frames, sliced along number of events in the recording. Good for generating a
                            pre-determined number of frames which might help with batching.
        overlap (float): overlap between frames defined either in time units, number of events or number of bins between 0 and 1.
        include_incomplete (bool): if True, includes overhang slice when time_window or spike_count is specified.
                                   Not valid for bin_count methods.
    """

    sensor_size: Tuple[int, int, int] = None
    time_window: Optional[float] = None
    event_count: Optional[int] = None
    n_time_bins: Optional[int] = None
    n_event_bins: Optional[int] = None
    overlap: float = 0
    include_incomplete: bool = False

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size

        return functional.to_frame_numpy(
            events=events,
            sensor_size=sensor_size,
            time_window=self.time_window,
            event_count=self.event_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
        )


def get_sensor_size(events: np.ndarray):
    return events["x"].max() + 1, events["y"].max() + 1, 2  # H,W,2


@dataclass(frozen=True)
class BackgroundActivityNoise:
    severity: int
    sensor_size: Tuple[int, int, int] = None

    def __call__(self, events):
        c = [.005, 0.01, 0.03, .10, .2][self.severity - 1]  # percentage of events to add in noise
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size
        n_noise_events = int(c * len(events))
        noise_events = np.zeros(n_noise_events, dtype=events.dtype)
        for channel in events.dtype.names:
            event_channel = events[channel]
            if channel == "x":
                low, high = 0, sensor_size[0]
            if channel == "y":
                low, high = 0, sensor_size[1]
            if channel == "p":
                low, high = 0, sensor_size[2]
            if channel == "t":
                low, high = events["t"].min(), events["t"].max()

            if channel == "p":
                noise_events[channel] = np.random.choice([True, False], size=n_noise_events)
            else:
                noise_events[channel] = np.random.uniform(
                    low=low, high=high, size=n_noise_events
                )
        events = np.concatenate((events, noise_events))
        new_events = events[np.argsort(events["t"])]
        new_events['p'] = events['p']

        return new_events


def get_frame_representation(sensor_size, timesteps):
    return transforms.Compose([
        # ToFrame(sensor_size=sensor_size, n_time_bins=timesteps),
        ToFrame(sensor_size=sensor_size, event_count=2500),
        TakeFrames(timesteps=timesteps),
        # transforms.Lambda(lambda x: (x > 0).astype(np.float32)),
        # transforms.Lambda(lambda x: torch.from_numpy(x))
        BinarizeFrame()
    ])


@dataclass(frozen=True)
class TakeFrames:
    timesteps: int

    def __call__(self, x):
        current_t = x.shape[0]
        # print(current_t)
        gap = int((current_t - self.timesteps) / 2)
        x = x[gap:gap + self.timesteps]
        return x


@dataclass(frozen=True)
class BinarizeFrame:
    def __call__(self, x):
        x = (x > 0).astype(np.float32)
        x = torch.from_numpy(x)
        return x


@dataclass(frozen=True)
class ConcatTimeChannels:
    def __call__(self, x):
        x = rearrange(x, 'frames polarity height width -> (frames polarity) height width')
        return x


# CutMix
@dataclass(frozen=True)
class CutMixEvents:
    generator: List
    num_mix: int = 2
    ratio: Tuple[float, float] = (0.2, 0.5)
    sensor_size: Tuple[int, int, int] = None

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size

        for _ in range(self.num_mix):
            mix, _ = random.choice(self.generator)

            bbx1, bby1, bbx2, bby2 = self._bbox(sensor_size[1], sensor_size[0])

            # filter image
            mask_events = (events['x'] >= bbx1) & (events['y'] >= bby1) & (events['x'] <= bbx2) & (events['y'] <= bby2)

            mask_mix = (mix['x'] >= bbx1) & (mix['y'] >= bby1) & (mix['x'] <= bbx2) & (mix['y'] <= bby2)

            # delete events of bbox
            events = np.delete(events, mask_events)  # remove events

            # add mix events in bbox
            events = np.concatenate((events, mix[mask_mix]))
            new_events = events[np.argsort(events["t"])]
            new_events['p'] = events['p']
            events = new_events

        return events

    def _bbox(self, H, W):
        ratio = random.uniform(self.ratio[0], self.ratio[1])

        cut_w = int(W * ratio)
        cut_h = int(H * ratio)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
