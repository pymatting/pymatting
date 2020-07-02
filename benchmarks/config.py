import os

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


def get_library_path(name):
    return os.path.join(BUILD_DIR, "pymatting", "solve_" + name + ".so")
