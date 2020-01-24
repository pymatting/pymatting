import numpy as np


def cg(
    A,
    b,
    x0=None,
    atol=0.0,
    rtol=1e-7,
    maxiter=10000,
    callback=None,
    M=None,
    reorthogonalize=False,
):
    """Solves a system of linear equations :math:`Ax=b` using conjugate gradient descent :cite:`hestenes1952methods`

    Parameters
    ----------
    A: scipy.sparse.csr_matrix
       Square matrix
    b: numpy.ndarray
       Vector describing the right-hand side of the system
    M: function
       Preconditioning matrix
    
    Returns
    -------
    x: numpy.ndarray
        Solution of the system
    """
    if M is None:
        precondition = lambda x: x
    elif callable(M):
        precondition = M
    else:
        precondition = M.dot(x)

    x = np.zeros_like(b) if x0 is None else x0.copy()

    norm_b = np.linalg.norm(b)

    if callable(A):
        r = b - A(x)
    else:
        r = b - A.dot(x)

    z = precondition(r)
    p = z.copy()
    rz = np.inner(r, z)

    for iteration in range(maxiter):
        r_old = r.copy()

        if callable(A):
            Ap = A(p)
        else:
            Ap = A.dot(p)

        alpha = rz / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        norm_r = np.linalg.norm(r)

        if callback is not None:
            callback(A, x, b, norm_b, r, norm_r)

        if norm_r < atol or norm_r < rtol * norm_b:
            return x

        z = precondition(r)

        if reorthogonalize:
            beta = np.inner(r - r_old, z) / rz
            rz = np.inner(r, z)
        else:
            beta = 1.0 / rz
            rz = np.inner(r, z)
            beta *= rz

        p *= beta
        p += z

    raise ValueError(
        "Conjugate gradient descent did not converge within %d iterations" % maxiter
    )
