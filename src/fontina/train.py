import argparse
import torch

from torchvision import datasets, transforms
from fontina.augmentation_utils import (
    get_deepfont_full_augmentations,
    get_random_square_patch_augmentation,
)
from fontina.augmented_dataset import AugmentedDataset

from fontina.config import load_config


def get_parser():
    parser = argparse.ArgumentParser(description="Fontina training")
    parser.add_argument(
        "-c",
        "--config",
        metavar="CONFIG-FILE",
        help="path to the configuration file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="out-data",
        metavar="OUTPUT-DIR",
        help="path to the directory that will contain the outputs",
    )
    return parser


def main():
    args = get_parser().parse_args()

    config = load_config(args.config)

    train_config = config["training"]

    all_data = datasets.ImageFolder(
        root=train_config["data_root"],
        # Important: albumentation can't set grayscale and output only one
        # channel, so do it here.
        transform=transforms.Grayscale(num_output_channels=1),
        target_transform=None,
    )
    full_data_num_classes = len(all_data.classes)

    print(f"All data:\n{all_data}\n\nClasses/ids:\n{all_data.class_to_idx}")

    train_set, test_set, val_set = torch.utils.data.random_split(
        all_data, [0.8, 0.10, 0.10]
    )

    train_set_processed = AugmentedDataset(
        train_set, full_data_num_classes, get_deepfont_full_augmentations()
    )
    test_set_processed = AugmentedDataset(
        test_set, full_data_num_classes, get_random_square_patch_augmentation()
    )
    validation_set_processed = AugmentedDataset(
        val_set, full_data_num_classes, get_random_square_patch_augmentation()
    )

    debug = True
    if debug:
        from torch.utils.data import DataLoader

        img, label = next(
            iter(
                DataLoader(train_set_processed, batch_size=1, num_workers=0, shuffle=True)
            )
        )
        print(f"Label index: {label[0]}")
        test = transforms.ToPILImage()(img[0])
        print(img[0].shape)
        test.save("test.png")


if __name__ == "__main__":
    main()
