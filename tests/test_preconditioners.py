import scipy.sparse.linalg
import numpy as np
import time
import json
import os
from pymatting import (
    load_image,
    show_images,
    trimap_split,
    cf_laplacian,
    make_linear_system,
    cg,
    jacobi,
    ichol,
    vcycle,
    ProgressCallback,
    CounterCallback,
)


def test_preconditioners():
    atol = 1e-6
    index = 1
    scale = 0.2
    max_error = 4.356

    name = f"GT{index:02d}"
    # print(name)

    image_dir = "data"

    image = load_image(
        f"{image_dir}/input_training_lowres/{name}.png", "rgb", scale, "bilinear"
    )
    trimap = load_image(
        f"{image_dir}/trimap_training_lowres/Trimap1/{name}.png",
        "gray",
        scale,
        "bilinear",
    )

    A, b = make_linear_system(cf_laplacian(image), trimap)

    preconditioners = [
        ("no", lambda A: None),
        ("jacobi", lambda A: jacobi(A)),
        ("icholt", lambda A: ichol(A)),
        ("vcycle", lambda A: vcycle(A, trimap.shape)),
    ]

    expected_iterations = {
        "no": 532,
        "jacobi": 250,
        "icholt": 3,
        "vcycle": 88,
    }

    for preconditioner_name, preconditioner in preconditioners:
        callback = CounterCallback()
        t0 = time.perf_counter()

        M = preconditioner(A)

        t1 = time.perf_counter()

        x = cg(A, b, M=M, atol=atol, rtol=0, maxiter=10000, callback=callback)

        t2 = time.perf_counter()

        r = b - A.dot(x)

        norm_r = np.linalg.norm(r)

        assert norm_r <= atol

        n_expected = expected_iterations[preconditioner_name]

        if callback.n > n_expected:
            print(
                "WARNING: Unexpected number of iterations. Expected %d, but got %d"
                % (n_expected, callback.n)
            )

        assert callback.n <= n_expected
