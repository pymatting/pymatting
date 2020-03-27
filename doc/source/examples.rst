Examples
========

We provide different examples at different levels of abstraction.

.. _example-simple:

Simple Example
---------------

This simple example is intended for application-oriented users.
All parameters were set beforehand and should work well on most images.
The :code:`cutout()` method employs closed-form alpha matting :cite:`levin2007closed` and multi-level foreground extraction :cite:`germer2020multilevel`.

.. code-block:: python

    from pymatting import cutout

    cutout(
       # input image path
       "../data/lemur.png",
       # input trimap path
       "../data/lemur_trimap.png",
       # output cutout path
       "lemur_cutout.png")


Advanced Example
----------------

The following example demonstrates the use of the :code:`estimate_alpha_cf()` method as well as the :code:`estimate_foreground_ml()` method.
Both methods can be easily replaced by other methods from the :code:`pymatting.alpha` and from the :code:`pymatting.foreground` module, respectively.
Parameters can be tweaked by passing them to the corresponding function calls.

.. code-block:: python

    from pymatting import *
    import numpy as np

    scale = 1.0

    image = load_image("../data/lemur.png", "RGB", scale, "box")
    trimap = load_image("../data/lemur_trimap.png", "GRAY", scale, "nearest")

    # estimate alpha from image and trimap
    alpha = estimate_alpha_cf(image, trimap)

    # make gray background
    background = np.zeros(image.shape)
    background[:, :] = [0.5, 0.5, 0.5]

    # estimate foreground from image and alpha
    foreground = estimate_foreground_ml(image, alpha)

    # blend foreground with background and alpha, less color bleeding
    new_image = blend(foreground, background, alpha)

    # save results in a grid
    images = [image, trimap, alpha, new_image]
    grid = make_grid(images)
    save_image("lemur_grid.png", grid)

    # save cutout
    cutout = stack_images(foreground, alpha)
    save_image("lemur_cutout.png", cutout)

    # just blending the image with alpha results in color bleeding
    color_bleeding = blend(image, background, alpha)
    grid = make_grid([color_bleeding, new_image])
    save_image("lemur_color_bleeding.png", grid)


Expert Example
--------------

The third example provides an insight how PyMatting is working under-the-hood. The matting Laplacian matrix :code:`L` and the system of linear equations :code:`A x = b` are constructed manually. The solution vector :code:`x` is the flattened alpha matte.
The alpha matte :code:`alpha` is then calculated by solving the linear system using the :code:`cg()` method. The convergence of the :code:`cg()` method is accelerated with a preconditioner using the :code:`ichol()` method.
This example is intended for developers and (future) contributors to demonstrate the implementation of the different alpha matting methods.

.. code-block:: python

    from pymatting import *
    import numpy as np
    import scipy.sparse

    scale = 1.0

    image = load_image("../data/lemur.png", "RGB", scale, "box")
    trimap = load_image("../data/lemur_trimap.png", "GRAY", scale, "nearest")

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

