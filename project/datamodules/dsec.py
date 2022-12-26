import os
import numpy as np
from tonic.dataset import Dataset


class DSEC(Dataset):
    sensor_size = (640, 480, 2)  # TODO
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(self, save_to="/datas/sandbox/", transform=None):
        super(DSEC, self).__init__(
            save_to=save_to,
            transform=transform,
            target_transform=None,
        )

        for dir_city in os.listdir(self.location_on_system):
            city_path = os.path.join(self.location_on_system, dir_city)
            for entry in os.listdir(city_path):
                if entry.endswith(".npy"):
                    self.data.append(os.path.join(city_path, entry))

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        input_path = self.data[index]
        events = np.load(input_path)

        if self.transform is not None:
            events = self.transform(events)

        return events

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.isdir(
            os.path.join(
                self.location_on_system, self.base_folder
            )  # check if directory exists
        ) and self._folder_contains_at_least_n_files_of_type(100, ".npy")
