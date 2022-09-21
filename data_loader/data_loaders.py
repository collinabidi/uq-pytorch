from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import torch
import h5py

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class H5Dataset(Dataset):
    """
    Credit: https://vict0rs.ch/2021/06/15/pytorch-h5/
    """
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self.archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        idx = 0
        for a, archive in enumerate(self.archives):
            for i in range(len(archive)):
                self.indices[idx] = (a, i)
        self._archives = None

    @property
    def archives(self):
        if self._archives is None: # Lazy loading
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives
    
    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive["data"]
        data = torch.from_numpy(dataset[:])
        label = archive["label"]
        labels = torch.from_numpy(label[:])

        return {"data":data, "labels":labels}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)

class CO2DataLoader(BaseDataLoader):
    """
    CO2 concentration dataset from Mauna Loa Observatory, Hawaii. Keeling et al., 2004
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = H5Dataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)