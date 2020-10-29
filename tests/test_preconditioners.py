import numpy as np
from pymatting import (
    load_image,
    cf_laplacian,
    make_linear_system,
    cg,
    jacobi,
    ichol,
    vcycle,
    CounterCallback,
)


def test_preconditioners():
    atol = 1e-6
    index = 1
    scale = 0.2

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
        "nearest",
    )

    A, b = make_linear_system(cf_laplacian(image), trimap)

    preconditioners = [
        ("no", lambda A: None),
        ("jacobi", lambda A: jacobi(A)),
        ("icholt", lambda A: ichol(A, max_nnz=500000)),
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

        M = preconditioner(A)

        x = cg(A, b, M=M, atol=atol, rtol=0, maxiter=10000, callback=callback)

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
