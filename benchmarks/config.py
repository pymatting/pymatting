import os
import numpy as np

BUILD_DIR = "/tmp/pymatting_benchmarks/"
IMAGE_DIR = "../data"
SOLVER_NAMES = [
    "cg_icholt",
    "pyamg",
    "mumps",
    "petsc",
    "amgcl",
    "umfpack",
    "superlu",
    "eigen_cholesky",
    "eigen_icholt",
]
ATOL = 1e-7
LABEL_NAMES = {
    "rw_laplacian": "Random Walk",
    "lbdm_laplacian": "Learning-Based $(r = 1)$",
    "knn_laplacian": "KNN",
    "cf_laplacian": "Closed-Form $(r = 1)$",
    "lkm_laplacian": "Large Kernel $(r = 10)$",
    # Uniform laplacian is just for testing, do not include in end results.
    # "uniform_laplacian": "Uniform",
}
INDICES = 1 + np.arange(27)
SCALES = np.sqrt(np.linspace(0.1, 1.0, 11))
# Uncomment for faster testing:
# SCALES = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]


def get_library_path(name):
    return os.path.join(BUILD_DIR, "pymatting", "solve_" + name + ".so")
