import numpy as np
import torch

from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    def __init__(self, dataset, num_classes, transform=None):
        self.dataset = dataset
        self.num_classes = num_classes
        # Transforms are albumentations
        self.transform = transform

    def __getitem__(self, index):
        raw_image = self.dataset[index][0]
        x = (
            self.transform(image=np.asarray(raw_image))["image"]
            if self.transform
            else raw_image
        )
        return x, torch.as_tensor(self.dataset[index][1], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)
