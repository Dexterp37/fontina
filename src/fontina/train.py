import argparse
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fontina.adobevfr_dataset import AdobeVFRDataset
from fontina.augmentation_utils import (
    get_deepfont_full_augmentations,
    get_random_square_patch_augmentation,
)
from fontina.augmented_dataset import AugmentedDataset

from fontina.config import load_config
from fontina.models.deepfont import DeepFont
from fontina.models.lightning_generate_callback import GenerateCallback
from fontina.models.lightning_wrappers import DeepFontAutoencoderWrapper, DeepFontWrapper


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
        "-r",
        "--resume-from",
        metavar="CHECKPOINT-FILE",
        help="path to the checkpoint file",
        required=False,
    )
    return parser


def load_and_split_data(train_config):
    all_data = datasets.ImageFolder(
        root=train_config["data_root"],
        # Important: albumentation can't set grayscale and output only one
        # channel, so do it here.
        transform=transforms.Grayscale(num_output_channels=1),
        target_transform=None,
    )
    full_data_num_classes = len(all_data.classes)

    split_ratios = (
        [
            train_config["train_ratio"],
            train_config["validation_ratio"],
            train_config["test_ratio"],
        ]
        if train_config["run_test_cycle"]
        else [
            train_config["train_ratio"],
            1.0 - train_config["train_ratio"],
        ]
    )
    splits = torch.utils.data.random_split(all_data, split_ratios)

    train_set_processed = AugmentedDataset(
        splits[0], full_data_num_classes, get_deepfont_full_augmentations()
    )
    validation_set_processed = AugmentedDataset(
        splits[1], full_data_num_classes, get_random_square_patch_augmentation()
    )
    test_set_processed = (
        AugmentedDataset(
            splits[2], full_data_num_classes, get_random_square_patch_augmentation()
        )
        if train_config["run_test_cycle"]
        else None
    )

    return (
        full_data_num_classes,
        train_set_processed,
        validation_set_processed,
        test_set_processed,
    )


def load_adobevfr_dataset(train_config):
    all_train_data = AdobeVFRDataset(
        f"{train_config['data_root']}/VFR_syn_train",
        "train",
        get_deepfont_full_augmentations(),
    )
    # Although the AdobeVFR dataset readme says that VFR_syn_val contains
    # the validation for the same classes as VFR_syn_train, that doens't
    # seem to be the case: the former contains 2383 classes, the latter
    # 4383. Instead of using it, let's split the rain set.
    splits = torch.utils.data.random_split(all_train_data, [0.95, 0.05])

    train_set_processed = AugmentedDataset(splits[0], all_train_data.num_labels, None)
    validation_set_processed = AugmentedDataset(
        splits[1], all_train_data.num_labels, None
    )

    """
    validation_set_processed = AdobeVFRDataset(
        f"{train_config['data_root']}/VFR_syn_val",
        "val",
        get_random_square_patch_augmentation(),
    )
    """
    test_set_processed = (
        AdobeVFRDataset(
            f"{train_config['data_root']}/VFR_real_test",
            "vfr_large",
            get_random_square_patch_augmentation(),
        )
        if train_config["run_test_cycle"]
        else None
    )

    return (
        all_train_data.num_labels,
        train_set_processed,
        validation_set_processed,
        test_set_processed,
    )


def main():
    args = get_parser().parse_args()

    config = load_config(args.config)

    train_config = config["training"]

    if "fixed_seed" in train_config:
        L.seed_everything(train_config["fixed_seed"])

    (
        full_data_num_classes,
        train_set_processed,
        validation_set_processed,
        test_set_processed,
    ) = (
        load_and_split_data(train_config)
        if train_config["dataset_type"] == "raw-images"
        else load_adobevfr_dataset(train_config)
    )

    batch_size = train_config["batch_size"]
    num_workers = max(train_config["num_workers"] // 3, 1)

    train_loader = DataLoader(
        train_set_processed,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_set_processed,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Try to make training a bit faster without sacrificing precision (as suggested
    # by lightning AI).
    torch.set_float32_matmul_precision("high")

    train_autoencoder = train_config["only_autoencoder"]
    model = None
    if train_autoencoder:
        model = DeepFontAutoencoderWrapper()
    else:
        autoenc_wrapper = DeepFontAutoencoderWrapper.load_from_checkpoint(
            train_config["scae_checkpoint_file"]
        )

        model = DeepFontWrapper(
            model=DeepFont(
                autoencoder=autoenc_wrapper.autoencoder,
                num_classes=full_data_num_classes,
            ),
            num_classes=full_data_num_classes,
            learning_rate=train_config["learning_rate"],
        )

    # Define the set of callbacks to use during the training process.
    training_callbacks = [
        EarlyStopping(monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=train_config["output_dir"],
            save_top_k=2,
            monitor="val_loss",
            filename="deepfont-{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}",
            save_last=True,
        ),
        ModelSummary(-1),
    ]

    if train_autoencoder:
        # When training the autoencoder,
        sample_index = 10 % len(train_set_processed)
        training_callbacks.append(
            GenerateCallback(
                train_set_processed[sample_index][0].unsqueeze(0), every_n_epochs=5
            ),
        )

    trainer = L.Trainer(
        # If ever doing gradient clipping here, remember that this
        # has an effect on training: it will take likely twice as
        # long to get to the same accuracy!
        # gradient_clip_algorithm="norm",
        # gradient_clip_val=1.0,
        max_epochs=train_config["epochs"],
        callbacks=training_callbacks,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=args.resume_from if args.resume_from else None,
    )

    # Test the model
    if "run_test_cycle" in train_config and train_config["run_test_cycle"]:
        test_loader = DataLoader(
            test_set_processed,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            persistent_workers=True,
        )
        trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
