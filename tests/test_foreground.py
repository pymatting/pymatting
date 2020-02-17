import numpy as np
import warnings
from pymatting import (
    load_image,
    estimate_foreground_cf,
    estimate_foreground_ml,
    trimap_split,
)


def test_foreground():
    try:
        from pymatting.foreground.estimate_foreground_ml_cupy import (
            estimate_foreground_ml_cupy,
        )
        from pymatting.foreground.estimate_foreground_ml_pyopencl import (
            estimate_foreground_ml_pyopencl,
        )

        methods = [
            estimate_foreground_ml,
            estimate_foreground_cf,
            estimate_foreground_ml_cupy,
            estimate_foreground_ml_pyopencl,
        ]
    except ImportError:
        methods = [
            estimate_foreground_ml,
            estimate_foreground_cf,
        ]
        warnings.warn(
            "Tests for GPU implementation skipped, because of missing packages."
        )

    max_mse = 0.022
    scale = 0.1
    image = load_image("data/lemur/lemur.png", "RGB", scale, "box")
    trimap = load_image("data/lemur/lemur_trimap.png", "GRAY", scale, "nearest")
    alpha = load_image("data/lemur/lemur_alpha.png", "GRAY", scale, "box")
    expected_foreground = load_image(
        "data/lemur/lemur_foreground.png", "RGB", scale, "box"
    )

    for estimate_foreground in methods:
        foreground, background = estimate_foreground(
            image, alpha, return_background=True
        )

        is_fg, is_bg, is_known, is_unknown = trimap_split(trimap, flatten=False)

        difference = np.abs(foreground - expected_foreground)
        weighted_difference = (alpha[:, :, np.newaxis] * difference)[is_unknown]

        mse = np.mean(weighted_difference)

        if mse > max_mse:
            print(
                "WARNING: %s - mean squared error threshold to expected foreground exceeded: %f"
                % (estimate_foreground.__name__, mse)
            )

        assert mse <= max_mse
