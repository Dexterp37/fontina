import fontina.augmentation_utils as au
import numpy as np


def test_resize_fixed_height():
    test_img = np.ones((70, 100), dtype=np.uint8)
    resized = au.resize_fixed_height(test_img, new_height=105)

    # Check for the expected height.
    assert resized.shape[0] == 105

    # Check that the ratio of the source image is kept.
    assert np.isclose(
        test_img.shape[0] / test_img.shape[1], resized.shape[0] / resized.shape[1]
    )


def test_split_patches():
    test_img = np.ones((70, 100), dtype=np.uint8)
    patches = au.split_patches_np(test_img, 50, False)

    # Check that we have the expected number of patches
    # of the expected size.
    assert len(patches) == 2
    assert all([p.shape[1] == 50 for p in patches])


def test_split_patches_partial():
    test_img = np.ones((70, 90), dtype=np.uint8)
    patches = au.split_patches_np(test_img, 50, False)

    # Even if the image width is shorter than 2 patches,
    # we still expect two of them: the split function will
    # fill part of the second one with full-white.
    assert len(patches) == 2
    assert all([p.shape[1] == 50 for p in patches])

    last_patch = patches[1]
    assert np.all(last_patch[:, 40:] == np.ones((70, 10), dtype=np.uint8) * 255)


def test_split_patches_drop():
    test_img = np.ones((70, 90), dtype=np.uint8)
    patches = au.split_patches_np(test_img, 50, True)

    # The image is too short for two patches and we explicitly
    # ask the function to drop the last partial patch.
    assert len(patches) == 1
    assert patches[0].shape[1] == 50


def test_pick_random_patch_constrained():
    # Craft an image with two 105x105 patches: one full black, one full white.
    test_img = np.ones((105, 110), dtype=np.uint8) * 255
    test_img[:, :105] = 0

    aug = au.PickRandomPatch(constrained_patches=True, always_apply=True)
    result = aug(image=test_img)["image"]

    assert np.all(np.equal(result.shape, [105, 105]))

    # It's either a full black or a full white patch, as we're constraining
    # the random choice into non-random patches.
    assert np.all(result == 255) or np.all(result == 0)


def test_pick_random_small_width():
    # Craft an image with one patch: its width being less than 105.
    test_img = np.ones((105, 80), dtype=np.uint8) * 255

    aug = au.PickRandomPatch(constrained_patches=True, always_apply=True)
    result = aug(image=test_img)["image"]

    # The returned patch must still be 105x105.
    assert np.all(np.equal(result.shape, [105, 105]))


def test_pick_random_exact_width():
    # Craft an image with one patch: its width being exactly 105.
    test_img = np.ones((105, 105), dtype=np.uint8) * 255

    aug = au.PickRandomPatch(constrained_patches=True, always_apply=True)
    result = aug(image=test_img)["image"]

    # The returned patch must still be 105x105.
    assert np.all(np.equal(result.shape, [105, 105]))
