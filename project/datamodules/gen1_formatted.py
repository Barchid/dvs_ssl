import os
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive, download_and_extract_archive
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Gen1Detection(Dataset):
    base_folder = "DailyAction-DVS"

    sensor_size = (304, 240, 2)  # DVS 128
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    classes = ["car", "pedestrian"]

    def __init__(self, save_to, subset="train", transform=None, target_transform=None):
        super(Gen1Detection, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.subset = subset

        if not self._check_exists():
            raise ValueError("The formatted Gen1 dataset does not exist.")

        if subset == "train":
            file_path = os.path.join(self.location_on_system, "train")
        elif subset == "val":
            file_path = os.path.join(self.location_on_system, "val")
        else:
            file_path = os.path.join(self.location_on_system, "test")

        for fil in file_path:
            if fil.endswith("_bbox.npy"):
                # self.targets.append(os.path.join(file_path, fil))
                self.data.append(
                    os.path.join(file_path, fil.replace("_bbox.npy", ".npy"))
                )

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        input_path = self.data[index]
        events = np.load(input_path)
        target = np.load(input_path.replace(".npy", "_bbox.npy"))
        target = self._to_gt(target)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.isdir(
            os.path.join(
                self.location_on_system, self.base_folder
            )  # check if directory exists
        ) and self._folder_contains_at_least_n_files_of_type(100, ".npy")

    def _to_gt(self, boxes):
        gt_boxes = torch.zeros((len(boxes), 4))
        gt_labels = torch.zeros(len(boxes), dtype=int)
        i = 0
        for boxe in boxes:
            x1 = boxe["x"]
            y1 = boxe["y"]
            x2 = np.clip(x1 + boxe["w"], 0, self.sensor_size[0])
            y2 = np.clip(y1 + boxe["h"], 0, self.sensor_size[1])
            label = boxe["class_id"]
            gt_boxes[i] = torch.tensor((x1, y1, x2, y2))
            gt_labels[i] = label
            i += 1
        return {"boxes": gt_boxes, "labels": gt_labels}
