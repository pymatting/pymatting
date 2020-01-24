from pymatting import *
import numpy as np
import scipy.sparse

scale = 1.0

image = load_image("../data/lemur/lemur.png", "RGB", scale, "box")
trimap = load_image("../data/lemur/lemur_trimap.png", "GRAY", scale, "nearest")

# height and width of trimap
h, w = trimap.shape[:2]

# calculate laplacian matrix
L = cf_laplacian(image)

# decompose trimap
is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

# constraint weight
lambda_value = 100.0

# build constraint pixel selection matrix
c = lambda_value * is_known
C = scipy.sparse.diags(c)

# build constraint value vector
b = lambda_value * is_fg

# build linear system
A = L + C

# build ichol preconditioner for faster convergence
A = A.tocsr()
A.sum_duplicates()
M = ichol(A)

# solve linear system with conjugate gradient descent
x = cg(A, b, M=M)

# clip and reshape result vector
alpha = np.clip(x, 0.0, 1.0).reshape(h, w)

save_image("lemur_alpha.png", alpha)
