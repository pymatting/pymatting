import numpy as np
import json
from pymatting import (
    load_image,
    trimap_split,
    make_linear_system,
    cg,
    ichol,
    LAPLACIANS,
    lkm_laplacian,
)


def main():
    indices = 1 + np.arange(27)
    scale = 1.0
    image_dir = "../data"
    info = {}

    laplacian_name = "lkm_laplacian"
    print(laplacian_name)

    errors = {}

    for index in indices:
        name = f"GT{index:02d}"
        print(name)

        image = load_image(
            f"{image_dir}/input_training_lowres/{name}.png", "rgb", scale, "bilinear"
        )
        trimap = load_image(
            f"{image_dir}/trimap_training_lowres/Trimap1/{name}.png",
            "gray",
            scale,
            "bilinear",
        )
        true_alpha = load_image(
            f"{image_dir}/gt_training_lowres/{name}.png", "gray", scale, "nearest"
        )

        L_matvec, diag_L = lkm_laplacian(image)

        is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

        atol = 1e-7 * np.sum(is_known)

        lambda_value = 100.0

        c = lambda_value * is_known
        b = lambda_value * is_fg

        inv_diag_A = 1.0 / (diag_L + c)

        def A_matvec(x):
            return L_matvec(x) + c * x

        def jacobi(x):
            return inv_diag_A * x

        x = cg(A_matvec, b, M=jacobi, atol=atol)

        alpha = np.clip(x, 0, 1).reshape(trimap.shape)

        difference_unknown = np.abs(alpha - true_alpha)[
            is_unknown.reshape(trimap.shape)
        ]

        error = np.linalg.norm(difference_unknown)

        errors[str(index)] = error

    info[laplacian_name] = errors

    for laplacian in LAPLACIANS:
        laplacian_name = laplacian.__name__
        print(laplacian_name)

        errors = {}

        for index in indices:
            name = f"GT{index:02d}"
            print(name)

            image = load_image(
                f"{image_dir}/input_training_lowres/{name}.png",
                "rgb",
                scale,
                "bilinear",
            )
            trimap = load_image(
                f"{image_dir}/trimap_training_lowres/Trimap1/{name}.png",
                "gray",
                scale,
                "bilinear",
            )
            true_alpha = load_image(
                f"{image_dir}/gt_training_lowres/{name}.png", "gray", scale, "nearest"
            )

            A, b = make_linear_system(laplacian(image), trimap)

            is_fg, is_bg, is_known, is_unknown = trimap_split(trimap, flatten=False)

            atol = 1e-7 * np.sum(is_known)

            preconditioner = {
                "knn_laplacian": jacobi,
                "rw_laplacian": jacobi,
                "cf_laplacian": ichol,
                "lbdm_laplacian": ichol,
                "uniform_laplacian": ichol,
            }[laplacian_name]

            x = cg(A, b, M=preconditioner(A), atol=atol)

            alpha = np.clip(x, 0, 1).reshape(trimap.shape)

            difference_unknown = np.abs(alpha - true_alpha)[is_unknown]

            error = np.linalg.norm(difference_unknown)

            errors[str(index)] = error

        info[laplacian_name] = errors

    with open("results/laplacians.json", "w") as f:
        print(info)
        json.dump(info, f, indent=4)


if __name__ == "__main__":
    main()
