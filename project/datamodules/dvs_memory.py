import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tonic
from torch.utils.data import Dataset
from torchvision import transforms, utils


class DvsMemory(Dataset):
    """Some Information about DvsMemory"""

    def __init__(self, dataset: tonic.Dataset, transform=None):
        super(DvsMemory, self).__init__()
        self.transform = transform
        self._load_dataset(dataset)

    def _load_dataset(self, dataset: tonic.Dataset):
        self.data = []
        self.targets = []
        
        print('Loading dataset in memory:')
        i = 0
        for events, target in dataset:
            print('Loading sample: ', i)
            self.data.append(events)
            self.targets.append(target)
        
        print('Sample loaded \n\n')

    def __getitem__(self, index):
        events = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)

        return events, target

    def __len__(self):
        return len(self.data)
