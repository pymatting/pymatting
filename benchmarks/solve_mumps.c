// mpicc main.c -ldmumps -o main && ./main

#include "mpi.h"
#include "dmumps_c.h"
#include <stdio.h>

int init_mpi(){
    return MPI_Init(NULL, NULL);
}

int finalize_mpi(){
    return MPI_Finalize();
}

int solve_mumps_coo(
    const double *coo_values,
    const int *i_inds,
    const int *j_inds,
    int nnz,
    double *x,
    int n,
    int is_symmetric,
    int print_info
){
    init_mpi();
    
    DMUMPS_STRUC_C id;
    MUMPS_INT myid, ierr;

    int error = 0;

    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    if (ierr != 0) return ierr;

    /* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
    id.comm_fortran = -987654;
    id.par = 1;
    id.sym = is_symmetric;
    id.job = -1;
    dmumps_c(&id);

    /* Define the problem on the host */
    if (myid == 0) {
        id.n = n;
        id.nz = nnz;
        id.irn = (int*)i_inds;
        id.jcn = (int*)j_inds;
        id.a = (double*)coo_values;
        id.rhs = x;
    }

    if (!print_info){
        // disable output
        for (int i = 0; i < 4; i++){
            id.icntl[i] = -1;
        }
    }

    id.job = 6;
    
    dmumps_c(&id);
    
    if (id.infog[0] < 0){
        printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n", myid, id.infog[0], id.infog[1]);
        error = 1;
    }

    id.job = -2;

    dmumps_c(&id);
    
    return error;
}
/*
int main(){
    int n = 3;
    int nnz = 3;
    const double coo_values[3] = {1.0, 2.0, 3.0};
    const int i_inds[3] = {1, 2, 3};
    const int j_inds[3] = {1, 2, 3};
    double x[3] = {4.0, 5.0, 6.0};
    int is_symmetric = 0;
    int print_info = 0;
    
    solve_mumps_coo(coo_values, i_inds, j_inds, nnz, x, n, is_symmetric, print_info);
    
    for (int i = 0; i < 3; i++){
        printf("%i - %f\n", i, x[i]);
    }
    
    return 0;
}
*/
