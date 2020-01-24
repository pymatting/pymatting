import numpy as np
import argparse
from pymatting import (
    load_image,
    estimate_foreground_cf,
    estimate_foreground_ml,
    show_images,
    trimap_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_gpu", help="exclude tests for GPU implementation", action="store_true"
    )
    args = parser.parse_args()

    if not args.no_gpu:
        from pymatting.foreground.estimate_foreground_ml_cupy import (
            estimate_foreground_ml_cupy,
        )
        from pymatting.foreground.estimate_foreground_ml_pyopencl import (
            estimate_foreground_ml_pyopencl,
        )

    max_mse = 0.022
    scale = 0.1
    image = load_image("data/lemur/lemur.png", "RGB", scale, "box")
    trimap = load_image("data/lemur/lemur_trimap.png", "GRAY", scale, "nearest")
    alpha = load_image("data/lemur/lemur_alpha.png", "GRAY", scale, "box")
    expected_foreground = load_image(
        "data/lemur/lemur_foreground.png", "RGB", scale, "box"
    )
    expected_background = load_image(
        "data/lemur/lemur_background.png", "RGB", scale, "box"
    )

    methods = [
        estimate_foreground_ml,
        estimate_foreground_cf,
    ]

    if not args.no_gpu:
        methods.append(estimate_foreground_ml_cupy)
        methods.append(estimate_foreground_ml_pyopencl)

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


if __name__ == "__main__":
    main()
