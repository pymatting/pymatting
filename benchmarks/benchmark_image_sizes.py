from pymatting import load_image, show_images, trimap_split
from pymatting import cf_laplacian, make_linear_system
from config import IMAGE_DIR, ATOL, SOLVER_NAMES, SCALES, INDICES
import scipy.sparse.linalg
import numpy as np
import threading
import multiprocessing
import psutil
import time
import json
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())

    return process.memory_info().rss


def log_memory_usage(memory_usage, log_interval=0.01):
    # Log current memory usage every few milliseconds
    thread = threading.currentThread()
    while thread.is_running:
        memory_usage.append(get_memory_usage())
        time.sleep(log_interval)


def build_solver(solver_name, A, Acsr, Acsc, Acoo, AL, b, atol, rtol):
    # Construct a solver from matrix A and vector b.

    if solver_name == "cg_icholt":
        from pymatting import cg, ichol

        M = ichol(A, discard_threshold=1e-3, shifts=[0.002])

        return lambda: cg(A, b, M=M, atol=atol, rtol=0)

    if solver_name == "pyamg":
        import pyamg
        from pymatting import cg

        M = pyamg.smoothed_aggregation_solver(A).aspreconditioner()

        return lambda: cg(Acsr, b, M=M, atol=atol, rtol=0)

    if solver_name == "mumps":
        from solve_mumps import solve_mumps_coo, init_mpi, finalize_mpi

        init_mpi()

        return lambda: solve_mumps_coo(AL.data, AL.row, AL.col, b, is_symmetric=True)

    if solver_name == "petsc":
        from solve_petsc import solve_petsc_coo, init_petsc, finalize_petsc

        init_petsc()

        return lambda: solve_petsc_coo(
            Acoo.data, Acoo.row, Acoo.col, b, atol=atol, gamg_threshold=0.1
        )

    if solver_name == "amgcl":
        from solve_amgcl import solve_amgcl_csr

        return lambda: solve_amgcl_csr(
            Acsr.data, Acsr.indices, Acsr.indptr, b, atol=atol, rtol=0
        )

    if solver_name == "umfpack":
        # Alternatively:
        # return lambda: scipy.sparse.linalg.spsolve(Acsc, b, use_umfpack=True)

        import scikits.umfpack

        return lambda: scikits.umfpack.spsolve(A, b)

    if solver_name == "superlu":
        # Alternatively:
        # scipy.sparse.linalg.spsolve(A, b, use_umfpack=False)

        return lambda: scipy.sparse.linalg.splu(Acsc).solve(b)

    if solver_name == "eigen_cholesky":
        from solve_eigen import solve_eigen_cholesky_coo

        return lambda: solve_eigen_cholesky_coo(Acoo.data, Acoo.row, Acoo.col, b)

    if solver_name == "eigen_icholt":
        from solve_eigen import solve_eigen_icholt_coo

        # Choose shift hust large enough to not fail for given images
        # (might fail with larger/different images)
        initial_shift = 5e-4

        return lambda: solve_eigen_icholt_coo(
            Acoo.data, Acoo.row, Acoo.col, b, rtol=rtol, initial_shift=initial_shift
        )

    raise ValueError(f"Solver {solver_name} does not exist.")


def run_solver_single_image(solver_name, scale, index):
    # Load images
    name = f"GT{index:02d}.png"

    image_path = os.path.join(IMAGE_DIR, "input_training_lowres", name)
    trimap_path = os.path.join(IMAGE_DIR, "trimap_training_lowres/Trimap1", name)

    image = load_image(image_path, "rgb", scale, "bilinear")
    trimap = load_image(trimap_path, "gray", scale, "nearest")

    # Create linear system
    L = cf_laplacian(image)

    A, b = make_linear_system(L, trimap)

    is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

    atol = ATOL * np.sum(is_known)
    rtol = atol / np.linalg.norm(b)

    # Compute various matrix representations
    Acsr = A.tocsr()
    Acsc = A.tocsc()
    Acoo = A.tocoo()

    AL = scipy.sparse.tril(Acoo)

    # Start memory usage measurement thread
    memory_usage = [get_memory_usage()]
    thread = threading.Thread(target=log_memory_usage, args=(memory_usage,))
    thread.is_running = True
    # All threads should die if the solver thread crashes so we can at least
    # carry on with the other solvers.
    thread.daemon = True
    thread.start()

    # Measure solver build time
    # Note that it is not easily possible to separate build time from solve time
    # for every solver, which is why only the sum of build_time and solve_time
    # should be compared for fairness.
    start_time = time.perf_counter()
    run_solver = build_solver(solver_name, A, Acsr, Acsc, Acoo, AL, b, atol, rtol)
    build_time = time.perf_counter() - start_time

    # Measure actual solve time
    start_time = time.perf_counter()
    x = run_solver()
    solve_time = time.perf_counter() - start_time

    # Stop memory usage measuring thread
    thread.is_running = False
    thread.join()

    # Compute relative error
    r = b - A.dot(x)
    norm_r = np.linalg.norm(r)

    # Store results
    h, w = trimap.shape

    result = dict(
        solver_name=str(solver_name),
        image_name=str(name),
        scale=float(scale),
        index=int(index),
        norm_r=float(norm_r),
        build_time=float(build_time),
        solve_time=float(solve_time),
        atol=float(atol),
        rtol=float(rtol),
        width=int(w),
        height=int(h),
        n_fg=int(np.sum(is_fg)),
        n_bg=int(np.sum(is_bg)),
        n_known=int(np.sum(is_known)),
        n_unknown=int(np.sum(is_unknown)),
        memory_usage=memory_usage,
    )

    print(result)

    # Ensure that everything worked as expected
    assert norm_r <= atol

    # Inspect alpha for debugging
    if 0:
        alpha = np.clip(x, 0, 1).reshape(h, w)
        show_images([alpha])

    return result


def run_solver(solver_name):
    print(f"Running {solver_name}")

    results = []

    # Dry run to ensure that everything has been loaded
    run_solver_single_image(solver_name, 0.1, 4)

    for scale in SCALES:
        for index in INDICES:
            result = run_solver_single_image(solver_name, scale, index)

            results.append(result)

    path = f"results/solver/{solver_name}.json"

    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def main():
    os.makedirs("results/solver/", exist_ok=True)

    # Run each solver in a new process
    for solver_name in SOLVER_NAMES:
        process = multiprocessing.Process(target=run_solver, args=(solver_name,))
        process.start()
        print("waiting for join...")
        process.join()
        print("joined")


if __name__ == "__main__":
    main()
