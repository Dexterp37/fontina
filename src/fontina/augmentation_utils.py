import albumentations as A
import numpy as np

from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image


def split_patches_np(img, step):
    height, width = img.shape

    patches = []
    for x in range(0, width, step):
        patches.append(img[0:height, x:x + step])

    # Fixup the last patch instead of dropping it because
    # its width is smaller than 105. When cropping with PIL and
    # the patch is smaller than the needed area, it gets filled
    # with black pixels. We should recolor them instead of discarding.
    available_width = width % step
    if available_width != 0:
        patches[-1] = np.append(
            patches[-1],
            np.full((height, step - available_width), 255, dtype="uint8"),
            axis=1,
        )

    return patches


class PickRandomPatch(ImageOnlyTransform):
    """
    Pick a random 105x105 patch.
    """

    def __init__(self, constrained_patches=True, always_apply=False, p=1.0) -> None:
        super(PickRandomPatch, self).__init__(always_apply, p)
        self.constrained_patches = constrained_patches

    def apply(self, img, **params):
        # If patches are constrained, then we split the
        # image into 105x105 boxes and pick one of the boxes.
        # Otherwise, we pick a random start coordinate and
        # build a box from there.
        if not self.constrained_patches:
            height, width = img.shape
            start_x = np.random.randint(0, width - 105)
            return img[0:105, start_x:start_x + 105]

        patches = split_patches_np(img, 105)
        patch_index = np.random.randint(0, len(patches))
        return patches[patch_index]


class VariableAspectRatio(ImageOnlyTransform):
    """
        The image, with heigh fi
    xed, is squeezed in width by a random
        ratio, drawn from a uniform distribution between a range.
    """

    def __init__(self, ratio_range, always_apply=False, p=1.0) -> None:
        super(VariableAspectRatio, self).__init__(always_apply, p)
        self.ratio_range = ratio_range

    def apply(self, img, **params):
        height, width = img.shape
        ratio = np.random.uniform(low=self.ratio_range[0], high=self.ratio_range[1])
        new_width = round(width * ratio)
        squeezed = Image.fromarray(img).resize(
            (new_width, height), Image.Resampling.LANCZOS
        )
        return np.array(squeezed)


class Squeezing(ImageOnlyTransform):
    """
    Add a "squeezing" operation = "we introduce a squeezing operation, that
    scales the width of the height-normalized image to be of a constant ratio relative
    to the height (2.5 in all our experiments). Note that the squeezing operation is
    equivalent to producing long rectangular input patches."
    """

    def __init__(self, squeeze_ratio, always_apply=False, p=1.0) -> None:
        super(Squeezing, self).__init__(always_apply, p)
        self.squeeze_ratio = squeeze_ratio

    def apply(self, img, **params):
        height, width = img.shape
        new_width = round(height * self.squeeze_ratio)
        squeezed = Image.fromarray(img).resize(
            (new_width, height), Image.Resampling.LANCZOS
        )
        return np.array(squeezed)


def get_deepfont_base_augmentations() -> A.Compose:
    return A.Compose(
        [
            # Parameters are taken from the paper.
            A.GaussNoise(var_limit=(3.0, 4.0), p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.5),
            # Images are expected to be grayscale and have a white background, so
            # use the 255 value as a filler.
            A.Affine(rotate=[-3, 3], cval=255, p=0.5),
            A.MultiplicativeNoise(multiplier=(0.3, 0.8), p=0.5),
        ]
    )


def get_deepfont_full_augmentations() -> A.Compose:
    return A.Compose([
        A.Sequential([
            get_deepfont_base_augmentations(),
            # From the deepfont paper, between 5/6 and 7/6.
            VariableAspectRatio(ratio_range=[0.83, 1.17], always_apply=True),
            Squeezing(squeeze_ratio=2.5, always_apply=True),
            PickRandomPatch(constrained_patches=False, always_apply=True),
            A.ToFloat(255, always_apply=True),
            ToTensorV2(),
        ]),
    ])


def get_random_square_patch_augmentation() -> A.Compose:
    return A.Compose(
        [
            PickRandomPatch(always_apply=True),
            A.ToFloat(255, always_apply=True),
            ToTensorV2(),
        ]
    )
