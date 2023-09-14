import numpy as np

from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    def __init__(self, dataset, num_classes, transform=None):
        self.dataset = dataset
        self.num_classes = num_classes
        # Transforms are albumentations
        self.transform = transform

    def __getitem__(self, index):
        raw_image = np.asarray(self.dataset[index][0])
        x = self.transform(image=raw_image)["image"] if self.transform else raw_image
        return x, self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)
