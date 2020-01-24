def jacobi(A):
    """
    Compute the Jacobi preconditioner function for the matrix A.
    
    Parameters
    -----------
    A: np.array
        Input matrix to compute the Jacobi preconditioner for.

    Returns
    ---------
    precondition_matvec: function
        Function which applies the Jacobi preconditioner to a vector
    """
    diagonal = A.diagonal()

    inverse_diagonal = 1.0 / diagonal

    def precondition_matvec(x):
        return x * inverse_diagonal

    return precondition_matvec
