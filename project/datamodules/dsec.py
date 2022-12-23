import os
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive, download_and_extract_archive
import struct


class DSecSSL(Dataset):
    base_folder = "TODO"

    sensor_size = (128, 128, 2)  # TODO
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(DSecSSL, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        file_path = os.path.join(self.location_on_system, self.base_folder)

        for act_dir in os.listdir(file_path):
            label = self.classes.index(act_dir)

            for file in os.listdir(os.path.join(file_path, act_dir)):
                if file.endswith("npy"):
                    self.targets.append(label)
                    self.data.append(os.path.join(file_path, act_dir, file))

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        input_path = self.data[index]
        events = np.load(input_path)

        target = self.targets[index]
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
