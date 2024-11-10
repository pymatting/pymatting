from PIL import Image
import numpy as np
import scipy.sparse
import os
import warnings
from functools import wraps


def apply_to_channels(single_channel_func):
    """Creates a new function which operates on each channel

    Parameters
    ----------
    single_channel_func: function
        Function that acts on a single color channel

    Returns
    -------
    channel_func: function
        The same function that operates on all color channels

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> from scipy.signal import convolve2d
    >>> single_channel_fun = lambda x: convolve2d(x, np.ones((3, 3)), 'valid')
    >>> multi_channel_fun = apply_to_channels(single_channel_fun)
    >>> I = np.random.rand(480, 320, 3)
    >>> multi_channel_fun(I).shape
    (478, 318, 3)
    """

    @wraps(single_channel_func)
    def multi_channel_func(image, *args, **kwargs):
        if len(image.shape) == 2:
            return single_channel_func(image, *args, **kwargs)
        else:
            shape = image.shape
            image = image.reshape(shape[0], shape[1], -1)

            result = np.stack(
                [
                    single_channel_func(image[:, :, c].copy(), *args, **kwargs)
                    for c in range(image.shape[2])
                ],
                axis=2,
            )

            return result.reshape(list(result.shape[:2]) + list(shape[2:]))

    return multi_channel_func


def vec_vec_dot(a, b):
    """Computes the dot product of two vectors.

    Parameters
    ----------
    a: numpy.ndarray
        First vector (if np.ndim(a) > 1 the function calculates the product for the two last axes)
    b: numpy.ndarray
        Second vector (if np.ndim(b) > 1 the function calculates the product for the two last axes)

    Returns
    -------
    product: scalar
        Dot product of `a` and `b`

    Example
    -------
    >>> import numpy as np
    >>> from pymatting import *
    >>> a = np.ones(2)
    >>> b = np.ones(2)
    >>> vec_vec_dot(a,b)
    2.0
    """
    return np.einsum("...i,...i->...", a, b)


def mat_vec_dot(A, b):
    """Calculates the matrix vector product for two arrays.

    Parameters
    ----------
    A: numpy.ndarray
        Matrix (if np.ndim(A) > 2 the function calculates the product for the two last axes)
    b: numpy.ndarray
        Vector (if np.ndim(b) > 1 the function calculates the product for the two last axes)

    Returns
    -------
    product: numpy.ndarray
        Matrix vector product of both arrays

    Example
    -------
    >>> import numpy as np
    >>> from pymatting import *
    >>> A = np.eye(2)
    >>> b = np.ones(2)
    >>> mat_vec_dot(A,b)
    array([1., 1.])
    """
    return np.einsum("...ij,...j->...i", A, b)


def vec_vec_outer(a, b):
    """Computes the outer product of two vectors

    a: numpy.ndarray
        First vector (if np.ndim(b) > 1 the function calculates the product for the two last axes)
    b: numpy.ndarray
        Second vector (if np.ndim(b) > 1 the function calculates the product for the two last axes)

    Returns
    -------
    product: numpy.ndarray
        Outer product of `a` and `b` as numpy.ndarray

    Example
    -------
    >>> import numpy as np
    >>> from pymatting import *
    >>> a = np.arange(1,3)
    >>> b = np.arange(1,3)
    >>> vec_vec_outer(a,b)
    array([[1, 2],
           [2, 4]])
    """
    return np.einsum("...i,...j", a, b)


def fix_trimap(trimap, lower_threshold=0.1, upper_threshold=0.9):
    """Fixes broken trimap :math:`T` by thresholding the values

    .. math::
        T^{\\text{fixed}}_{ij}=
        \\begin{cases}
            0,&\\text{if } T_{ij}<\\text{lower\\_threshold}\\\\
            1,&\\text{if }T_{ij}>\\text{upper\\_threshold}\\\\
            0.5, &\\text{otherwise}.\\\\
        \\end{cases}


    Parameters
    ----------
    trimap: numpy.ndarray
        Possibly broken trimap
    lower_threshold: float
        Threshold used to determine background pixels, defaults to 0.1
    upper_threshold: float
        Threshold used to determine foreground pixels, defaults to 0.9

    Returns
    -------
    fixed_trimap: numpy.ndarray
        Trimap having values in :math:`\\{0, 0.5, 1\\}`

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> trimap = np.array([0,0.1, 0.4, 0.9, 1])
    >>> fix_trimap(trimap, 0.2, 0.8)
    array([0. , 0. , 0.5, 1. , 1. ])
    """
    if lower_threshold < 0 or lower_threshold > 1:
        raise ValueError("Invalid lower threshold")

    if upper_threshold < 0 or upper_threshold > 1:
        raise ValueError("Invalid upper threshold")

    if lower_threshold > upper_threshold:
        raise ValueError("Lower threshold must be smaller than upper threshold")

    is_bg = trimap < lower_threshold
    is_fg = trimap > upper_threshold
    fixed = 0.5 * np.ones_like(trimap)
    fixed[is_fg] = 1
    fixed[is_bg] = 0
    return fixed


def isiterable(obj):
    """Checks if an object is iterable

    Parameters
    ----------
    obj: object
        Object to check

    Returns
    -------
    is_iterable: bool
        Boolean variable indicating whether the object is iterable

    Example
    -------
    >>> from pymatting import *
    >>> l = []
    >>> isiterable(l)
    True
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _resize_pil_image(image, size, resample="bicubic"):
    filters = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "box": Image.BOX,
        "hamming": Image.HAMMING,
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST,
        "none": Image.NEAREST,
    }

    if not isiterable(size):
        size = (int(image.width * size), int(image.height * size))

    image = image.resize(size, filters[resample.lower()])

    return image


def load_image(path, mode=None, size=None, resample="box"):
    """This function can be used to load an image from a file.

    Parameters
    ----------
    path: str
        Path of image to load.
    mode: str
        Can be \"GRAY\", \"RGB\" or something else (see PIL.convert())

    Returns
    -------
    image: numpy.ndarray
        Loaded image
    """

    image = Image.open(path)

    if mode is not None:
        mode = mode.upper()
        mode = "L" if mode == "GRAY" else mode
        image = image.convert(mode)

    if size is not None:
        image = _resize_pil_image(image, size, resample)

    image = np.array(image) / 255.0

    return image


def save_image(path, image, make_directory=True):
    """Given a path, save an image there.

    Parameters
    ----------
    path: str
        Where to save the image.
    image: numpy.ndarray, dtype in [np.uint8, np.float32, np.float64]
        Image to save.
        Images of float dtypes should be in range [0, 1].
        Images of uint8 dtype should be in range [0, 255]
    make_directory: bool
        Whether to create the directories needed for the image path.
    """
    assert image.dtype in [np.uint8, np.float32, np.float64]

    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    if make_directory:
        directory, _ = os.path.split(path)
        if len(directory) > 0:
            os.makedirs(directory, exist_ok=True)

    Image.fromarray(image).save(path)


def to_rgb8(image):
    """Convertes an image to rgb8 color space

    Parameters
    ----------
    image: numpy.ndarray
        Image to convert

    Returns
    -------
    image: numpy.ndarray
        Converted image with same height and width as input image but with three color channels
    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> I = np.eye(2)
    >>> to_rgb8(I)
    array([[[255, 255, 255],
            [  0,   0,   0]],
           [[  0,   0,   0],
            [255, 255, 255]]], dtype=uint8)
    """
    assert len(image.shape) in [2, 3]
    assert image.dtype in [np.uint8, np.float32, np.float64]

    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    if len(image.shape) == 2:
        return np.stack([image] * 3, axis=2)

    if image.shape[2] == 1:
        return np.concatenate([image] * 3, axis=2)

    elif image.shape[2] == 3:
        return image

    elif image.shape[2] == 4:
        return image[:, :, :3]

    raise ValueError("Invalid image shape:", image.shape)


def make_grid(images, nx=None, ny=None, dtype=None):
    """Plots a grid of images.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of images to plot
    nx: int
        Number of rows
    ny: int
        Number of columns
    dtype: type
        Data type of output array

    Returns
    -------
    grid: numpy.ndarray
       Grid of images with datatype `dtype`
    """
    for image in images:
        if image is not None:
            assert image.dtype in [np.float32, np.float64]

    n = len(images)

    if n == 0:
        return

    if nx is None and ny is None:
        nx = int(np.ceil(np.sqrt(n)))
        ny = (n + nx - 1) // nx

    elif nx is None:
        ny = (n + nx - 1) // nx

    elif ny is None:
        nx = (n + ny - 1) // ny

    shapes = [image.shape for image in images if image is not None]

    h = max(shape[0] for shape in shapes)
    w = max(shape[1] for shape in shapes)
    d = max([shape[2] for shape in shapes if len(shape) > 2], default=1)

    if d > 1:
        for i, image in enumerate(images):
            if image is not None:
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]

                if image.shape[2] == 1:
                    image = np.concatenate([image] * d, axis=2)

                if image.shape[2] == 3 and d == 4:
                    image = stack_images(
                        image, np.ones(image.shape[:2], dtype=image.dtype)
                    )

                images[i] = image

    if dtype is None:
        dtype = next(image.dtype for image in images if image is not None)

    result = np.zeros((h * ny, w * nx, d), dtype=dtype)

    for y in range(ny):
        for x in range(nx):
            i = x + y * nx

            if i >= len(images):
                break

            image = images[i]

            if image is not None:
                image = image.reshape(image.shape[0], image.shape[1], -1)

                result[
                    y * h : y * h + image.shape[0], x * w : x * w + image.shape[1]
                ] = image

    if result.shape[2] == 1:
        result = result[:, :, 0]

    return result


def show_images(images):
    """Plot grid of images.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of images to plot
    height : int, matrix
        Height in pixels the output grid, defaults to 512

    """
    grid = make_grid(images)

    grid = np.clip(grid * 255, 0, 255).astype(np.uint8)
    grid = Image.fromarray(grid)
    grid.show()


def trimap_split(trimap, flatten=True, bg_threshold=0.1, fg_threshold=0.9):
    """This function splits the trimap into foreground pixels, background pixels, and unknown pixels.

    Foreground pixels are pixels where the trimap has values larger than or equal to `fg_threshold` (default: 0.9).
    Background pixels are pixels where the trimap has values smaller than or equal to `bg_threshold` (default: 0.1).
    Pixels with other values are assumed to be unknown.

    Parameters
    ----------
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times w`
    flatten: bool
        If true np.flatten is called on the trimap

    Returns
    -------
    is_fg: numpy.ndarray
        Boolean array indicating which pixel belongs to the foreground
    is_bg: numpy.ndarray
        Boolean array indicating which pixel belongs to the background
    is_known: numpy.ndarray
        Boolean array indicating which pixel is known
    is_unknown: numpy.ndarray
        Boolean array indicating which pixel is unknown
    bg_threshold: float
        Pixels with smaller trimap values will be considered background.
    fg_threshold: float
        Pixels with larger trimap values will be considered foreground.


    Example
    -------
    >>> import numpy as np
    >>> from pymatting import *
    >>> trimap = np.array([[1,0],[0.5,0.2]])
    >>> is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)
    >>> is_fg
    array([ True, False, False, False])
    >>> is_bg
    array([False,  True, False, False])
    >>> is_known
    array([ True,  True, False, False])
    >>> is_unknown
    array([False, False,  True,  True])
    """
    if flatten:
        trimap = trimap.flatten()

    min_value = trimap.min()
    max_value = trimap.max()

    if min_value < 0.0:
        warnings.warn(
            "Trimap values should be in [0, 1], but trimap.min() is %s." % min_value,
            stacklevel=3,
        )

    if max_value > 1.0:
        warnings.warn(
            "Trimap values should be in [0, 1], but trimap.max() is %s." % min_value,
            stacklevel=3,
        )

    if trimap.dtype not in [np.float32, np.float64]:
        warnings.warn(
            "Unexpected trimap.dtype %s. Are you sure that you do not want to use np.float32 or np.float64 instead?"
            % trimap.dtype,
            stacklevel=3,
        )

    is_fg = trimap >= fg_threshold
    is_bg = trimap <= bg_threshold

    if is_bg.sum() == 0:
        raise ValueError(
            "Trimap did not contain background values (values <= %f)" % bg_threshold
        )

    if is_fg.sum() == 0:
        raise ValueError(
            "Trimap did not contain foreground values (values >= %f)" % fg_threshold
        )

    is_known = is_fg | is_bg
    is_unknown = ~is_known

    return is_fg, is_bg, is_known, is_unknown


def sanity_check_image(image):
    """Performs a sanity check for input images. Image values should be in the
    range [0, 1], the `dtype` should be `np.float32` or `np.float64` and the
    image shape should be `(?, ?, 3)`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h \\times w \\times 3`

    Example
    -------
    >>> import numpy as np
    >>> from pymatting import check_image
    >>> image = (np.random.randn(64, 64, 2) * 255).astype(np.int32)
    >>> sanity_check_image(image)
    __main__:1: UserWarning: Expected RGB image of shape (?, ?, 3), but image.shape is (64, 64, 2).
    __main__:1: UserWarning: Image values should be in [0, 1], but image.min() is -933.
    __main__:1: UserWarning: Image values should be in [0, 1], but image.max() is 999.
    __main__:1: UserWarning: Unexpected image.dtype int32. Are you sure that you do not want to use np.float32 or np.float64 instead?

    """

    if len(image.shape) != 3 or image.shape[2] != 3:
        warnings.warn(
            "Expected RGB image of shape (?, ?, 3), but image.shape is %s."
            % str(image.shape),
            stacklevel=3,
        )

    min_value = image.min()
    max_value = image.max()

    if min_value < 0.0:
        warnings.warn(
            "Image values should be in [0, 1], but image.min() is %s." % min_value,
            stacklevel=3,
        )

    if max_value > 1.0:
        warnings.warn(
            "Image values should be in [0, 1], but image.max() is %s." % max_value,
            stacklevel=3,
        )

    if image.dtype not in [np.float32, np.float64]:
        warnings.warn(
            "Unexpected image.dtype %s. Are you sure that you do not want to use np.float32 or np.float64 instead?"
            % image.dtype,
            stacklevel=3,
        )


def blend(foreground, background, alpha):
    """This function composes a new image for given foreground image, background image and alpha matte.

    This is done by applying the composition equation

    .. math::
        I = \\alpha F + (1-\\alpha)B.

    Parameters
    ----------
    foreground: numpy.ndarray
        Foreground image
    background: numpy.ndarray
        Background image
    alpha: numpy.ndarray
        Alpha matte

    Returns
    -------
    image: numpy.ndarray
        Composed image as numpy.ndarray

    Example
    -------
    >>> from pymatting import *
    >>> foreground = load_image("data/lemur/lemur_foreground.png", "RGB")
    >>> background = load_image("data/lemur/beach.png", "RGB")
    >>> alpha = load_image("data/lemur/lemur_alpha.png", "GRAY")
    >>> I = blend(foreground, background, alpha)
    """
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]

    return alpha * foreground + (1 - alpha) * background


def stack_images(*images):
    """This function stacks images along the third axis.
    This is useful for combining e.g. rgb color channels or color and alpha channels.

    Parameters
    ----------
    *images: numpy.ndarray
        Images to be stacked.

    Returns
    -------
    image: numpy.ndarray
        Stacked images as numpy.ndarray

    Example
    -------
    >>> from pymatting.util.util import stack_images
    >>> import numpy as np
    >>> I = stack_images(np.random.rand(4,5,3), np.random.rand(4,5,3))
    >>> I.shape
    (4, 5, 6)
    """
    images = [
        (image if len(image.shape) == 3 else image[:, :, np.newaxis])
        for image in images
    ]
    return np.concatenate(images, axis=2)


def row_sum(A):
    """Calculate the sum of each row of a matrix

    Parameters
    ----------
    A: np.ndarray or scipy.sparse.spmatrix
        Matrix to sum rows of

    Returns
    -------
    row_sums: np.ndarray
        Vector of summed rows

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> A = np.random.rand(2,2)
    >>> A
    array([[0.62750946, 0.12917617],
           [0.8599449 , 0.5777254 ]])
    >>> row_sum(A)
    array([0.75668563, 1.4376703 ])
    """
    row_sums = A.dot(np.ones(A.shape[1], A.dtype))

    return row_sums


def normalize_rows(A, threshold=0.0):
    """Normalize the rows of a matrix

    Rows with sum below threshold are left as-is.

    Parameters
    ----------
    A: scipy.sparse.spmatrix
        Matrix to normalize
    threshold: float
        Threshold to avoid division by zero

    Returns
    -------
    A: scipy.sparse.spmatrix
        Matrix with normalized rows

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> A = np.arange(4).reshape(2,2)
    >>> normalize_rows(A)
    array([[0. , 1. ],
           [0.4, 0.6]])
    """
    row_sums = row_sum(A)

    # Prevent division by zero.
    row_sums[row_sums < threshold] = 1.0

    row_normalization_factors = 1.0 / row_sums

    D = scipy.sparse.diags(row_normalization_factors)

    A = D.dot(A)

    return A


def grid_coordinates(width, height, flatten=False):
    """Calculates image pixel coordinates for an image with a specified shape

    Parameters
    ----------
    width: int
        Width of the input image
    height: int
        Height of the input image
    flatten: bool
        Whether the array containing the coordinates should be flattened or not, defaults to False

    Returns
    -------
    x: numpy.ndarray
        x coordinates
    y: numpy.ndarray
        y coordinates

    Example
    -------
    >>> from pymatting import *
    >>> x, y = grid_coordinates(2,2)
    >>> x
    array([[0, 1],
           [0, 1]])
    >>> y
    array([[0, 0],
           [1, 1]])
    """
    if flatten:
        x = np.tile(np.arange(width), height)
        y = np.repeat(np.arange(height), width)
    else:
        x = np.arange(width)
        y = np.arange(height)

        x, y = np.meshgrid(x, y)

    return x, y


def sparse_conv_matrix_with_offsets(width, height, kernel, dx, dy):
    """Calculates a convolution matrix that can be applied to a vectorized image

    Additionally, this function allows to specify which pixels should be used for the convoltion, i.e.

    .. math:: \\left(I * K\\right)_{ij} = \\sum_k K_k I_{i+{\\Delta_y}_k,j+{\\Delta_y}_k},

    where :math:`K` is the flattened convolution kernel.

    Parameters
    ----------
    width: int
        Width of the input image
    height: int
        Height of the input image
    kernel: numpy.ndarray
        Convolutional kernel
    dx: numpy.ndarray
        Offset in x direction
    dy: nunpy.ndarray
        Offset in y direction

    Returns
    -------
    M: scipy.sparse.csr_matrix
        Convolution matrix
    """
    weights = np.asarray(kernel).flatten()
    count = len(weights)
    n = width * height

    i_inds = np.zeros(n * count, dtype=np.int32)
    j_inds = np.zeros(n * count, dtype=np.int32)
    values = np.zeros(n * count, dtype=np.float64)

    k = 0
    x, y = grid_coordinates(width, height, flatten=True)
    for dx2, dy2, weight in zip(dx, dy, weights):
        x2 = np.clip(x + dx2, 0, width - 1)
        y2 = np.clip(y + dy2, 0, height - 1)
        i_inds[k : k + n] = x + y * width
        j_inds[k : k + n] = x2 + y2 * width
        values[k : k + n] = weight
        k += n

    A = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))

    return A


def sparse_conv_matrix(width, height, kernel):
    """Calculates a convolution matrix that can be applied to a vectorized image

    Parameters
    ----------
    width: int
        Width of the input image
    height: int
        Height of the input image
    kernel: numpy.ndarray
        Convolutional kernel

    Returns
    -------
    M: scipy.sparse.csr_matrix
        Convolution matrix

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> sparse_conv_matrix(3,3,np.ones((3,3)))
    <9x9 sparse matrix of type '<class 'numpy.float64'>'
    with 49 stored elements in Compressed Sparse Row format>
    """
    kh, kw = kernel.shape
    x, y = grid_coordinates(kw, kh, flatten=True)
    x -= kw // 2
    y -= kh // 2

    return sparse_conv_matrix_with_offsets(width, height, kernel, x, y)


def weights_to_laplacian(W, normalize=True, regularization=0.0):
    """Calculates the random walk normalized Laplacian matrix from the weight matrix

    Parameters
    ----------
    W: numpy.ndarray
        Array of weights
    normalize: bool
        Whether the rows of W should be normalized to 1, defaults to True
    regularization: float
        Regularization strength, defaults to 0, i.e. no regularizaion

    Returns
    -------
    L: scipy.sparse.spmatrix
        Laplacian matrix

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> weights_to_laplacian(np.ones((4,4)))
    matrix([[ 0.75, -0.25, -0.25, -0.25],
            [-0.25,  0.75, -0.25, -0.25],
            [-0.25, -0.25,  0.75, -0.25],
            [-0.25, -0.25, -0.25,  0.75]])
    """
    if normalize:
        W = normalize_rows(W)

    d = regularization + row_sum(W)
    D = scipy.sparse.diags(d)

    L = D - W

    return L


def normalize(values):
    """Normalizes an array such that all values are between 0 and 1

    Parameters
    ----------
    values: numpy.ndarray
        Array to normalize

    Returns
    -------
    result: numpy.ndarray
        Normalized array

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> normalize(np.array([0, 1, 3, 10]))
    array([0. , 0.1, 0.3, 1. ])
    """
    values = np.asarray(values)
    a = values.min()
    b = values.max()
    return (values - a) / (b - a)


def div_round_up(x, n):
    """Divides a number x by another integer n and rounds up the result

    Parameters
    ----------
    x: int
        Numerator
    n: int
        Denominator

    Returns
    -------
    result: int
        Result

    Example
    -------
    >>> from pymatting import *
    >>> div_round_up(3,2)
    2
    """
    return (x + n - 1) // n


def remove_background_bicolor(image, fg_color, bg_color):
    """Remove background from image with at most two colors.
    Might not work if image has more than two colors.

    Parameters
    ----------
    image: numpy.ndarray
        RGB input image
    fg_color: numpy.ndarray
        RGB Foreground color
    bg_color: numpy.ndarray
        RGB Background color

    Returns
    -------
    output: numpy.ndarray
        RGBA output image

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> image = np.random.rand(480, 320, 3)
    >>> fg_color = np.random.rand(3)
    >>> bg_color = np.random.rand(3)
    >>> output = remove_background_bicolor(image, fg_color, bg_color)
    >>> print(output.shape)
    (480, 320, 4)
    """

    fg_bg = fg_color - bg_color

    # Project image colors onto (fg, bg) color line and clamp to [0, 1] to obtain alpha
    u = np.sum((image - bg_color) * fg_bg, axis=-1) / np.sum(np.square(fg_bg))
    alpha = np.clip(u, 0.0, 1.0)

    # Unmix colors by solving
    # image = actual_color * alpha + bg_color * (1 - alpha)
    # for actual_color
    a = alpha[:, :, None]

    actual_color = np.divide(
        (image - (1 - a) * bg_color), a, out=np.zeros_like(image), where=a != 0.0
    )

    actual_color = np.clip(actual_color, 0.0, 1.0)

    output = np.dstack([actual_color, alpha])

    return output
