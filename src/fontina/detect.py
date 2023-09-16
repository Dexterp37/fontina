import argparse
import numpy as np
import PIL
import torch
import torch.nn.functional as F

from fontina.augmentation_utils import (
    get_deepfont_feature_enhancement,
    get_random_square_patch,
)

from fontina.models.deepfont import DeepFont, DeepFontAutoencoder
from fontina.models.lightning_wrappers import DeepFontWrapper


def get_parser():
    parser = argparse.ArgumentParser(description="Fontina detect")
    parser.add_argument(
        "-w",
        "--weights",
        metavar="WEIGHTS-FILE",
        help="path to the weights/checkpoint file",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num-classes",
        type=int,
        help="the number of classes",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="INPUT-FILE",
        help="path to the input image file",
        required=True,
    )
    return parser


def predict(model: DeepFontWrapper, img):
    # Apply feature enhancement.
    enhancement_pipeline = get_deepfont_feature_enhancement()

    # TODO: we need 3 different versions of the same sample and
    # average the outputs (as required by the paper).
    all_soft_preds = []
    for _ in range(3):
        enhanced_img = enhancement_pipeline(image=np.asarray(img))["image"]

        patch_sampler = get_random_square_patch()
        patches = [patch_sampler(image=enhanced_img)["image"] for _ in range(5)]
        inputs = torch.tensor(np.asarray(patches))

        preds = model(inputs.cuda())
        soft_preds = F.softmax(preds, dim=1)
        all_soft_preds.append(soft_preds)

    return torch.cat(all_soft_preds).mean(0).argmax()


def main():
    args = get_parser().parse_args()

    model = DeepFontWrapper.load_from_checkpoint(
        args.weights,
        model=DeepFont(autoencoder=DeepFontAutoencoder(), num_classes=args.num_classes),
        num_classes=args.num_classes,
    )

    def resize_fixed_height(img, new_height=105):
        # From the paper: height is fixed to 105 pixels, width is scaled
        # to keep aspect ratio.
        width, height = img.size
        new_width = round(new_height * width / height)
        return img.resize((new_width, new_height), PIL.Image.LANCZOS)

    raw_img = PIL.Image.open(args.input).convert("L")
    img = resize_fixed_height(raw_img)

    predicted_class = predict(model, img)
    print(f"Final Predicted label {predicted_class}")


if __name__ == "__main__":
    main()
