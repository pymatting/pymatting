from pymatting import load_image, show_images, trimap_split
from pymatting import cf_laplacian, make_linear_system
from collections import defaultdict
import scipy.sparse.linalg
import numpy as np
import threading
import psutil
import time
import json
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    
    return process.memory_info().rss

def main():
    solvers = []
    
    if 0:
        from pymatting import cg, ichol
        
        def solve_cg_icholt(A, b, atol):
            M = ichol(A, discard_threshold=1e-3, shifts=[0.002])
            return cg(A, b, M=M, atol=atol, rtol=0)
        
        solvers.append(("cg_icholt", lambda: solve_cg_icholt(Acsc, b, atol=atol)))
    
    if 1:
        import pyamg
        from pymatting import cg
        
        def solve_pyamg(A, b, atol):
            M = pyamg.smoothed_aggregation_solver(A).aspreconditioner()
            return cg(A, b, M=M, atol=atol, rtol=0)

        solvers.append(("pyamg", lambda: solve_pyamg(Acsr, b, atol=atol)))
    
    if 0:
        from solve_mumps import solve_mumps_coo, init_mpi, finalize_mpi
        init_mpi()
        
        def solve_mumps(A, b):
            return solve_mumps_coo(A.data, A.row, A.col, b, is_symmetric=True)

        solvers.append(("mumps", lambda: solve_mumps(AL, b)))
    
    if 1:
        from solve_petsc import solve_petsc_coo, init_petsc, finalize_petsc
        init_petsc()
    
        def solve_petsc(A, b, atol):
            return solve_petsc_coo(A.data, A.row, A.col, b, atol=atol, gamg_threshold=0.1)

        solvers.append(("petsc", lambda: solve_petsc(Acoo, b, atol=atol)))

    if 1:
        from solve_amgcl import solve_amgcl_csr
    
        def solve_amgcl(A, b, atol):
            return solve_amgcl_csr(A.data, A.indices, A.indptr, b, atol=atol, rtol=0)

        solvers.append(("amgcl", lambda: solve_amgcl(Acsr, b, atol=atol)))
    
    if 0:
        solvers.append(("umfpack", lambda: scipy.sparse.linalg.spsolve(Acsc, b, use_umfpack=True)))
    
    if 0:
        def solve_superLU(A, b):
            # Usually the same as:
            # scipy.sparse.linalg.spsolve(A, b, use_umfpack=False)
            return scipy.sparse.linalg.splu(A).solve(b)

        solvers.append(("superlu", lambda: solve_superLU(Acsc, b)))
    
    if 1:
        from solve_eigen import solve_eigen_cholesky_coo
        
        def solve_eigen_cholesky(A, b):
            return solve_eigen_cholesky_coo(A.data, A.row, A.col, b)

        solvers.append(("eigen_cholesky", lambda: solve_eigen_cholesky(Acoo, b)))
    
    if 0:
        from solve_eigen import solve_eigen_icholt_coo
        
        def solve_eigen_icholt(A, b, rtol):
            # just large enough to not fail for given images (might fail with larger/different images)
            initial_shift = 2e-4
            return solve_eigen_icholt_coo(A.data, A.row, A.col, b, rtol=rtol, initial_shift=initial_shift)

        solvers.append(("eigen_icholt", lambda: solve_eigen_icholt(Acoo, b, rtol=rtol)))
    
    indices = np.arange(27)
    scales = np.sqrt(np.linspace(0.1, 1.0, 11))
    
    #indices = np.int32([1, 2])
    #scales = [0.1]
    
    atol0 = 1e-7
    
    image_dir = "data"
    
    for solver_name, solver in solvers:
        results = []
        
        for scale in scales:
            for index in indices + 1:
                
                name = f"GT{index:02d}"
                image = load_image(f"{image_dir}/input_training_lowres/{name}.png", "rgb", scale, "bilinear")
                trimap = load_image(f"{image_dir}/trimap_training_lowres/Trimap1/{name}.png", "gray", scale, "bilinear")
                true_alpha = load_image(f"{image_dir}/gt_training_lowres/{name}.png", "gray", scale, "nearest")
                
                L = cf_laplacian(image)
                
                A, b = make_linear_system(L, trimap)
                
                is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)
                
                atol = atol0 * np.sum(is_known)
                rtol = atol / np.linalg.norm(b)
                
                Acsr = A.tocsr()
                Acsc = A.tocsc()
                Acoo = A.tocoo()
                AL = scipy.sparse.tril(Acoo)
                
                def log_memory_usage(memory_usage):
                    while running:
                        memory_usage.append(get_memory_usage())
                        time.sleep(0.01)
                
                memory_usage = [get_memory_usage()]
                running = True
                thread = threading.Thread(target=log_memory_usage, args=(memory_usage,))
                thread.start()
                
                start_time = time.perf_counter()
                
                x = solver()
                
                elapsed_time = time.perf_counter() - start_time
                
                running = False
                thread.join()
                
                r = b - A.dot(x)
                norm_r = np.linalg.norm(r)
                
                h, w = trimap.shape
                
                result = dict(
                    solver_name = str(solver_name),
                    scale = float(scale),
                    index = int(index),
                    norm_r = float(norm_r),
                    t = float(elapsed_time),
                    atol = float(atol),
                    rtol = float(rtol),
                    width = int(w),
                    height = int(h),
                    n_fg = int(np.sum(is_fg)),
                    n_bg = int(np.sum(is_bg)),
                    n_known = int(np.sum(is_known)),
                    n_unknown = int(np.sum(is_unknown)),
                    memory_usage = memory_usage,
                )
                
                print(result)
                
                assert(norm_r <= atol)

                if 0:
                    alpha = np.clip(x, 0, 1).reshape(h, w)
                    show_images([alpha])
                
                results.append(result)

        path = f"benchmarks/results/solver/{solver_name}/results.json"
        
        dir_name = os.path.split(path)[0]
        if len(dir_name) > 0:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
