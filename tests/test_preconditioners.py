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
    scale = 0.125

    image_path = "data/lemur/lemur.png"
    trimap_path = "data/lemur/lemur_trimap.png"

    image = load_image(image_path, "rgb", scale, "bilinear")
    trimap = load_image(trimap_path, "gray", scale, "nearest")

    A, b = make_linear_system(cf_laplacian(image), trimap)

    preconditioners = [
        ("none", lambda A: None),
        ("jacobi", lambda A: jacobi(A)),
        ("icholt", lambda A: ichol(A, max_nnz=500000)),
        ("vcycle", lambda A: vcycle(A, trimap.shape)),
    ]

    expected_iterations = {
        "none": 378,
        "jacobi": 197,
        "icholt": 3,
        "vcycle": 48,
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
