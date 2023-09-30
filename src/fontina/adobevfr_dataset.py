import numpy as np
import io
import torch

from PIL import Image
from torch.utils.data import Dataset


def read_label(label_file: str):
    with open(label_file, "rb") as f:
        # We need to cast to `int32` as PyTorch 2.0.1 does not yet
        # support collating `uin16`, `uint32` in data loaders. See
        # https://github.com/pytorch/pytorch/issues/58734
        return np.frombuffer(f.read(), np.uint32, -1).astype("int32")


class AdobeVFRDataset(Dataset):
    def __init__(self, bcf_path: str, dataset_type: str, transform=None):
        self.bcf_filename = f"{bcf_path}/{dataset_type}.bcf"
        self.labels = read_label(f"{bcf_path}/{dataset_type}.label")
        self.num_labels = len(np.unique(self.labels))
        # Transforms are albumentations
        self.transform = transform

        self._load_bcf_meta()

    def __getitem__(self, index):
        binary_image = self._get_bcf_entry_by_index(index)
        pil_image = Image.open(io.BytesIO(binary_image)).convert("L")
        raw_image = np.array(pil_image, dtype="uint8")
        x = self.transform(image=raw_image)["image"] if self.transform else raw_image
        # We need to cast to `torch.long` to prevent errors such as
        # "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'.
        return x, torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self._bcf_offsets) - 1

    def _load_bcf_meta(self):
        with open(self.bcf_filename, "rb") as f:
            # Read the size of the section with files?
            size = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            # Must cast for some reason, otherwise this yields a float64.
            file_sizes = np.frombuffer(f.read(int(8 * size)), dtype=np.uint64)
            self._bcf_offsets = np.append(np.uint64(0), np.add.accumulate(file_sizes))

    def _get_bcf_entry_by_index(self, i):
        with open(self.bcf_filename, "rb") as f:
            f.seek((len(self._bcf_offsets) * 8 + self._bcf_offsets[i]).astype("uint64"))
            return f.read(self._bcf_offsets[i + 1] - self._bcf_offsets[i])
