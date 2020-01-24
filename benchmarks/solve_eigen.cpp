#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers> 

using namespace Eigen;

typedef SparseMatrix<double, ColMajor> MySparseMatrix;

extern "C" {

int solve_eigen_icholt_coo(
    const double *coo_data,
    const int *row,
    const int *col,
    int nnz,
    const double *b,
    double *x,
    int n,
    double rtol,
    double initial_shift
){
    MySparseMatrix A(n, n);
    
    std::vector<Triplet<double> > triplets(nnz);
    
    for (int k = 0; k < nnz; k++){
        triplets[k] = Triplet<double>(row[k], col[k], coo_data[k]);
    }
    
    A.setFromTriplets(triplets.begin(), triplets.end());
    
    A.makeCompressed();
    
    VectorXd b_temp(n);
    
    for (int k = 0; k < n; k++){
        b_temp[k] = b[k];
    }
    
    typedef IncompleteCholesky<double> Preconditioner;
    ConjugateGradient<MySparseMatrix, Lower, Preconditioner> solver;
    
    solver.preconditioner().setInitialShift(initial_shift);
    
    solver.setTolerance(rtol);
    
    solver.compute(A);
    
    VectorXd x_temp = solver.solve(b_temp);

    for (int k = 0; k < n; k++){
        x[k] = x_temp[k];
    }
    
    return 0;
}

int solve_eigen_cholesky_coo(
    const double *coo_data,
    const int *row,
    const int *col,
    int nnz,
    const double *b,
    double *x,
    int n,
    double rtol,
    double initial_shift
){
    MySparseMatrix A(n, n);
    
    std::vector<Triplet<double> > triplets(nnz);
    
    for (int k = 0; k < nnz; k++){
        triplets[k] = Triplet<double>(row[k], col[k], coo_data[k]);
    }
    
    A.setFromTriplets(triplets.begin(), triplets.end());
    
    A.makeCompressed();
    
    VectorXd b_temp(n);
    
    for (int k = 0; k < n; k++){
        b_temp[k] = b[k];
    }
    
    SimplicialLDLT<MySparseMatrix> solver;
    
    solver.analyzePattern(A);
    
    solver.factorize(A);
    
    VectorXd x_temp = solver.solve(b_temp);

    for (int k = 0; k < n; k++){
        x[k] = x_temp[k];
    }
    
    return 0;
}

}
