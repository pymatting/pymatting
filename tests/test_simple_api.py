from pymatting import *
import os
import numpy as np


def test_cutout():
    output_path = "test_cutout.png"

    cutout("data/test/test.png", "data/test/test_trimap.png", output_path)

    cutout_img = load_image(output_path)

    os.remove(output_path)

    foreground = cutout_img[:, :, :3]
    alpha = cutout_img[:, :, 3]

    true_alpha = load_image("data/test/test_alpha.png", "GRAY")
    true_foreground = load_image("data/test/test_foreground.png", "RGB")

    error_alpha = np.linalg.norm(alpha - true_alpha)
    error_foreground = np.linalg.norm(
        alpha[:, :, np.newaxis] * np.abs(foreground - true_foreground)
    )

    assert error_alpha < 3.1
    assert error_foreground < 4
