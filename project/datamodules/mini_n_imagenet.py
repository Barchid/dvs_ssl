import os
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class MiniNImageNet(Dataset):

    base_folder = "DVS-Lip"

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    classes = [0] * 100 # 100 classes

    def __init__(self, save_to, train=True, transform=None, target_transform=None):
        super(MiniNImageNet, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.url = self.base_url
        self.folder_name = os.path.join(
            self.base_folder, "train" if self.train else "test"
        )

        self.train_path = os.path.join(self.location_on_system, "extracted_train")
        self.val_path = os.path.join(self.location_on_system, "extracted_val")
        

        self.classes = os.listdir(self.val_path)
        print(self.classes)
        print(len(self.classes))
        
        file_path = self.train_path if self.train else self.val_path
        # get all npz filename
        for class_dir in os.listdir(file_path):
            label = self.classes.index(class_dir)

            for file in os.listdir(os.path.join(file_path, class_dir)):
                if file.endswith("npz"):
                    self.targets.append(label)
                    self.data.append(os.path.join(file_path, class_dir, file))

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        orig_events = np.load(self.data[index])

        events = np.empty(orig_events.shape, dtype=self.dtype)
        events["x"] = orig_events["x"]
        events["y"] = orig_events["y"]
        events["t"] = orig_events["t"]
        events["p"] = orig_events["p"]

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
                self.location_on_system, self.folder_name
            )  # check if directory exists
        ) and self._folder_contains_at_least_n_files_of_type(100, ".npz")