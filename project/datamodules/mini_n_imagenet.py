import os
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class MiniNImageNet(Dataset):

    base_folder = "DVS-Lip"

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    classes = [
        "n01739381",
        "n01755581",
        "n01530575",
        "n01644900",
        "n01756291",
        "n01665541",
        "n01632777",
        "n01697457",
        "n01770393",
        "n01689811",
        "n01667114",
        "n01687978",
        "n01806143",
        "n01820546",
        "n01729322",
        "n01748264",
        "n01498041",
        "n01855032",
        "n01843065",
        "n01629819",
        "n01768244",
        "n01688243",
        "n01582220",
        "n01749939",
        "n01744401",
        "n01817953",
        "n01496331",
        "n01608432",
        "n01735189",
        "n01443537",
        "n01729977",
        "n01532829",
        "n01494475",
        "n01824575",
        "n01537544",
        "n01773157",
        "n01664065",
        "n01675722",
        "n01774750",
        "n01616318",
        "n01798484",
        "n01692333",
        "n01806567",
        "n01682714",
        "n01641577",
        "n01531178",
        "n01704323",
        "n01491361",
        "n01592084",
        "n01534433",
        "n01829413",
        "n01818515",
        "n01796340",
        "n01833805",
        "n01795545",
        "n01843383",
        "n01774384",
        "n01770081",
        "n01740131",
        "n01514859",
        "n01514668",
        "n01614925",
        "n01847000",
        "n01694178",
        "n01775062",
        "n01742172",
        "n01440764",
        "n01560419",
        "n01622779",
        "n01753488",
        "n01773797",
        "n01630670",
        "n01737021",
        "n01728920",
        "n01677366",
        "n01580077",
        "n01667778",
        "n01776313",
        "n01693334",
        "n01807496",
        "n01773549",
        "n01797886",
        "n01855672",
        "n01631663",
        "n01644373",
        "n01734418",
        "n01751748",
        "n01518878",
        "n01558993",
        "n01632458",
        "n01669191",
        "n01484850",
        "n01728572",
        "n01601694",
        "n01695060",
        "n01819313",
        "n01828970",
        "n01784675",
        "n01698640",
        "n01685808",
    ]

    def __init__(self, save_to, train=True, transform=None, target_transform=None):
        super(MiniNImageNet, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train

        self.train_path = os.path.join(self.location_on_system, "extracted_train")
        self.val_path = os.path.join(self.location_on_system, "extracted_val")

        self.classes = os.listdir(self.val_path)
        print(self.classes)
        print(len(self.classes))
        exit()

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
