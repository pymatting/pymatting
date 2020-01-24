#include <assert.h>
#include <stdio.h>
#include <vector>

#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>

#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/spai1.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/relaxation/ilu0.hpp>

extern "C" {

int solve_amgcl_csr(
    const double *csr_values,
    const int *csr_indices,
    const int *csr_indptr,
    int nnz,
    const double *b,
    double *x,
    int n,
    double atol,
    double rtol,
    int maxiter
){
    std::vector<int> ptr(csr_indptr, csr_indptr + n + 1);
    std::vector<int> col(csr_indices, csr_indices + nnz);
    std::vector<double> val(csr_values, csr_values + nnz);
    std::vector<double> rhs(b, b + n);
    
    typedef amgcl::backend::builtin<double> Backend;
    
    typedef amgcl::make_solver<
        // amg preconditioner
        amgcl::amg<
            Backend,
            // coarsener
            
            //amgcl::coarsening::ruge_stuben,
            amgcl::coarsening::smoothed_aggregation,
            
            // smoothing method
            amgcl::relaxation::spai0
            // slower
            //amgcl::relaxation::gauss_seidel
            //amgcl::relaxation::spai1
            //amgcl::relaxation::chebyshev
            //amgcl::relaxation::ilu0
            //amgcl::relaxation::ilut
        >,
        // solver:
        //amgcl::solver::bicgstab<Backend>
        amgcl::solver::cg<Backend>
    > Solver;

    Solver::params params;
    params.solver.tol = rtol;
    params.solver.abstol = atol;
    params.solver.maxiter = maxiter;
    
    Solver solve(std::tie(n, ptr, col, val), params);
    
    std::vector<double> result(n, 0.0);
    
    int iters = -1;
    double error;
    std::tie(iters, error) = solve(rhs, result);
    
    for (int i = 0; i < n; i++){
        x[i] = result[i];
    }
    
    return iters;
}
 
}
