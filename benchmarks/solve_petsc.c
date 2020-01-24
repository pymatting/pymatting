/*
static char help[] =
    "Compile and run:\n"
    "mpicc main.c -I/usr/lib/petsc/include -o main -lpetsc -lm && /usr/bin/time -v mpiexec -n 1 ./main -ksp_converged_reason -ksp_final_residual";
*/
// https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCType.html
// reasonable pc_types:
// -pc_type icc/hypre/mg/gamg
/*
mpiexec -n 1 ./main -ksp_atol 1e-8 -ksp_rtol 0 -pc_type gamg -pc_gamg_threshold 0.15

mpicc main.c -I/usr/lib/petsc/include -o main -lpetsc -lm && mpiexec -n 1 ./main -ksp_atol 1e-8 -ksp_rtol 0 -log_view -info -ksp_type cg -pc_type gamg
mpicc main.c -I/usr/lib/petsc/include -o main -lpetsc -lm && mpiexec -n 1 ./main -ksp_atol 1e-8 -ksp_rtol 0 -pc_type gamg
mpicc main.c -I/usr/lib/petsc/include -o main -lpetsc -lm && mpiexec -n 1 ./main -ksp_atol 0 -ksp_rtol 1e-6 -pc_type gamg -pc_gamg_threshold 0.2 -pc_gamg_square_graph 0
mpicc main.c -I/usr/lib/petsc/include -o main -lpetsc -lm && mpiexec -n 1 ./main -ksp_atol 0 -ksp_rtol 1e-6 -pc_type gamg -pc_gamg_threshold 0.2 -pc_gamg_square_graph 0 -info | grep GAMG
*/

#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

double sec(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

int init_petsc(){
    int argc = 0;
    char **args = NULL;
    PetscErrorCode ierr;
    PetscMPIInt rank,size;
    ierr = PetscInitialize(&argc,&args,NULL,NULL);if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    return ierr;
}

int finalize_petsc(){
    return PetscFinalize();
}

int solve_petsc_coo(
    const double *coo_data,
    const int *row,
    const int *col,
    int nnz,
    const double *rhs,
    double *result,
    int n,
    double atol,
    double rtol,
    double gamg_threshold,
    int maxiter
){
    int *i_inds = (int*)row;
    int *j_inds = (int*)col;
    double *values = (double*)coo_data;
    
    int *indices = (int*)malloc(n*sizeof(*indices));
    
    for (int i = 0; i < n; i++){
        indices[i] = i;
    }

    Mat A;
    Vec x,b,r;
    KSP ksp;
    PC pc;
    PetscErrorCode ierr;

    ierr = MatCreateSeqAIJFromTriple(PETSC_COMM_WORLD, n, n, i_inds, j_inds, values, &A, nnz, 0);
    CHKERRQ(ierr);

    // create right hand side and solution
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
    
    ierr = VecSet(x,0.0);CHKERRQ(ierr);
    ierr = VecSet(b,0.0);CHKERRQ(ierr);
    ierr = VecSet(r,0.0);CHKERRQ(ierr);
    
    ierr = VecSetValues(b, n, indices, rhs, INSERT_VALUES);CHKERRQ(ierr);

    // inizialize Krylov subspace solver
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    // should be preconditioned conjugate gradient descent
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    // set preconditioner
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCGAMG);CHKERRQ(ierr);
    ierr = PCGAMGSetThreshold(pc,&gamg_threshold,1);CHKERRQ(ierr);
    // set tolerance
    ierr = KSPSetTolerances(ksp, rtol, atol, PETSC_DEFAULT, maxiter);CHKERRQ(ierr);
    // use |A x - b| as termination criterion instead of preconditioned residual
    ierr = KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);CHKERRQ(ierr);
    // solve linear system
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

#ifdef DEBUG_PETSC
    ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    
    int n_iter;
    ierr = KSPGetIterationNumber(ksp, &n_iter);CHKERRQ(ierr);
    printf("petsc converged after %d iterations\n", n_iter);
    
    KSPConvergedReason converged_reason = KSP_CONVERGED_ITERATING;
    ierr = KSPGetConvergedReason(ksp, &converged_reason);CHKERRQ(ierr);
    printf("petsc converged reason: %i\n", (int)converged_reason);
#endif

    ierr = VecGetValues(x, n, indices, result); CHKERRQ(ierr);
  
    free(indices);

    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    
    return ierr;
}
/*
typedef struct {
    int i;
    int j;
    double value;
} Triple;

int compare(const void *p, const void *q){
    const Triple *a = p;
    const Triple *b = q;
    
    if (a->i != b->i) return a->i < b->i ? -1 : 1;
    if (a->j != b->j) return a->j < b->j ? -1 : 1;
    
    return 0;
}

int main(){
    const int n = 100;
    const int more = 1000;
    const int nnz = n + more;
    
    double x[n];
    double b[n];
    double rhs[n];
    
    Triple *triples = malloc(sizeof(*triples) * nnz);
    
    uint8_t *used = calloc(n, n);
    
    for (int i = 0; i < n; i++){
        double value = rand() / (double)RAND_MAX + n;
        
        triples[i] = (Triple){i, i, value};
        
        rhs[i] = rand() / (double)RAND_MAX;
        
        x[i] = 0.0;
        b[i] = 0.0;
        used[i + i * n] = 1;
    }
    
    for (int k = n; k < n + more; k++){
        while (1){
            int i = rand() % n;
            int j = rand() % n;
            
            if (used[i + j * n]) continue;
            
            used[i + j * n] = 1;
            
            double value = rand() / (double)RAND_MAX;
            
            triples[k] = (Triple){i, i, value};
            break;
        }
    }
    
    qsort(triples, nnz, sizeof(*triples), compare);
    
    int i_inds[nnz];
    int j_inds[nnz];
    double values[nnz];
    
    for (int k = 0; k < nnz; k++){
        const Triple *t = &triples[k];
        
        i_inds[k] = t->i;
        j_inds[k] = t->j;
        values[k] = t->value;
    }
    
    double atol = 1e-5;
    double rtol = 1e-5;
    double gamg_threshold = 0.15;
    
    int err = solve_petsc_coo(values, i_inds, j_inds, nnz, rhs, x, n, atol, rtol, gamg_threshold, 10000);
    
    if (err != 0) return err;
    
    for (int k = 0; k < nnz; k++){
        int i = i_inds[k];
        int j = j_inds[k];
        
        double value = values[k];
        
        b[i] += value * x[j];
    }
    
    double error = 0.0;
    
    for (int i = 0; i < n; i++){
        double d = b[i] - rhs[i];
        
        error += d * d;
    }
    
    printf("sum (Ax - b)^2 = %.50f\n", error);
    
    return 0;
}
*/
