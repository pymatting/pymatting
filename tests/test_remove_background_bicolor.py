import numpy as np
from pymatting import load_image, remove_background_bicolor, blend


def test_remove_background_bicolor():
    image_path = "data/test/logo.png"
    expected_path = "data/test/logo_rgba.png"

    fg_color = np.array([0x86, 0xFF, 0x1A]) / 255.0
    bg_color = np.array([0xF4, 0x7F, 0x32]) / 255.0

    image = load_image(image_path, "RGB")
    expected = load_image(expected_path, "RGBA")

    output = remove_background_bicolor(image, fg_color, bg_color)

    new_bg = np.array([0.1, 0.2, 0.3])

    composite = blend(output[:, :, :3], new_bg, output[:, :, 3])
    expected_composite = blend(expected[:, :, :3], new_bg, expected[:, :, 3])

    difference = np.abs(composite - expected_composite)

    assert np.max(difference) < 0.1

    mse = np.mean(np.square(composite - expected_composite))

    assert mse < 1e-4
