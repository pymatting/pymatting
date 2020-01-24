from pymatting.laplacian.cf_laplacian import cf_laplacian
from pymatting.laplacian.knn_laplacian import knn_laplacian
from pymatting.laplacian.lkm_laplacian import lkm_laplacian
from pymatting.laplacian.rw_laplacian import rw_laplacian
from pymatting.laplacian.lbdm_laplacian import lbdm_laplacian
from pymatting.laplacian.uniform_laplacian import uniform_laplacian
from pymatting.laplacian.laplacian import make_linear_system

LAPLACIANS = [
    cf_laplacian,
    knn_laplacian,
    lbdm_laplacian,
    rw_laplacian,
    uniform_laplacian,
]

LAPLACIAN_OPERATORS = [
    lkm_laplacian,
]
