import os
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive, download_and_extract_archive
import struct


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
                self.targets.append(os.path.join(file_path, fil))
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
        target = np.load(input_path.replace('.npy', '_bbox.npy'))
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
        
    def _to_gt(target: dict):
        return {
            'boxes': [],
            'labels': []
        }

def to_gt(target):
    """
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
    """
    
    return {}
    
def getDVSeventsDavis(file, numEvents=1e10, startTime=0):
    """DESCRIPTION: This function reads a given aedat file and converts it into four lists indicating
                     timestamps, x-coordinates, y-coordinates and polarities of the event stream.

    Args:
        file: the path of the file to be read, including extension (str).
        numEvents: the maximum number of events allowed to be read (int, default value=1e10).
        startTime: the start event timestamp (in microseconds) where the conversion process begins (int, default value=0).
    Return:
        ts: list of timestamps in microseconds.
        x: list of x-coordinates in pixels.
        y: list of y-coordinates in pixels.`
        pol: list of polarities (0: on -> off, 1: off -> on).
    """
    # print("\ngetDVSeventsDavis function called \n")
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY

    # print("Reading in at most", str(numEvents))

    triggerevent = int("400", 16)
    polmask = int("800", 16)
    xmask = int("003FF000", 16)
    ymask = int("7FC00000", 16)
    typemask = int("80000000", 16)
    typedvs = int("00", 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []
    pol = []
    numeventsread = 0

    length = 0
    aerdatafh = open(file, "rb")
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    # print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from(">II", tmp)
        ad = abs(ad)
        if tm >= startTime:
            if (ad & typemask) == typedvs:
                xo = sizeX - 1 - float((ad & xmask) >> xshift)
                yo = float((ad & ymask) >> yshift)
                polo = 1 - float((ad & polmask) >> polshift)
                if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                    x.append(xo)
                    y.append(yo)
                    pol.append(polo)
                    ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    # print("Total number of events read =", numeventsread)
    # print("Total number of DVS events returned =", len(ts))

    return ts, x, y, pol
