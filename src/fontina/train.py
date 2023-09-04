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
from fontina.augmentation_utils import (
    get_deepfont_full_augmentations,
    get_random_square_patch_augmentation,
)
from fontina.augmented_dataset import AugmentedDataset

from fontina.config import load_config
from fontina.models.deepfont import DeepFont, DeepFontAutoencoder
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
    return parser


def main():
    args = get_parser().parse_args()

    config = load_config(args.config)

    train_config = config["training"]

    if "fixed_seed" in train_config:
        L.seed_everything(train_config["fixed_seed"])

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

    batch_size = train_config["batch_size"]
    num_workers = train_config["num_workers"] // 3

    # num_workers must be 0 to avoid "process exited unexpectedly"
    train_loader = DataLoader(
        train_set_processed,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set_processed,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True,
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
        # Load the trained autoencoder.
        checkpoint = torch.load(train_config["scae_checkpoint_file"])
        autoenc_model = DeepFontAutoencoder()

        # update keys by dropping `autoencoder.`
        autoenc_weights = checkpoint["state_dict"]
        for key in list(autoenc_weights):
            autoenc_weights[key.replace("autoencoder.", "")] = autoenc_weights.pop(key)

        autoenc_model.load_state_dict(autoenc_weights)

        model = DeepFontWrapper(
            model=DeepFont(autoencoder=autoenc_model, num_classes=full_data_num_classes),
            num_classes=full_data_num_classes,
            learning_rate=train_config["learning_rate"],
        )

    # train model
    trainer = L.Trainer(
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        max_epochs=train_config["epochs"],
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=train_config["output_dir"],
                save_top_k=2,
                monitor="val_loss",
                filename="deepfont-{epoch:02d}-{val_loss:.4f}-{val_accuracy:.2f}",
            ),
            ModelSummary(-1),
        ],
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    # Test the model
    if "run_test_cycle" in train_config and train_config["run_test_cycle"]:
        trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
