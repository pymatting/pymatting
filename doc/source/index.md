PyMatting is a Python library to cut out things from an image.
More specifically, PyMatting estimates an alpha matte from an input image and an input trimap.
The trimap is used to specify which subject from the image should be extracted. It consists of three parts:

1. White, which is 100 % foreground.
2. Black, which is 100 % background.
3. Gray, which is either foreground, background or a mix of both. PyMatting fills in the gray area of the trimap to compute the alpha matte.

.. html::
    <div class="figure">
        <img src="/figures/lemur_at_the_beach.png">
        <div class="caption">Figure 1: Input image, input trimap, estimated alpha and extracted foreground. </div>
    </div>

## Installation

To install PyMatting, simply run:

   git clone https://github.com/pymatting/pymatting
   cd pymatting
   pip3 install .

## Testing

Run the tests from the main directory:

   python3 tests/download_images.py
   pip3 install -r requirements_tests.txt
   pytest

A warning will be thrown if PyOpenCL or CuPy are not available.

## Requirements

* numpy>=1.16.0
* pillow>=5.2.0
* numba>=0.44.0
* scipy>=1.1.0

## Additional Requirements (for GPU support)

* cupy-cuda90>=6.5.0 or similar
* pyopencl>=2019.1.2

## Alpha Matting

For an image :math:`I` with foreground pixels :math:`F` and background :math:`B` the alpha matting problem aims to determine opacities :math:`\alpha`, such that the equality

.. math::
   I = \alpha F +(1-\alpha)B

holds. This problem is inherently ill-posed since for each pixel we have three equations with seven unknown variables. The alpha matte :math:`\alpha` determine how much a pixel contributes to the foreground and how much to the background of an image.

After estimating the alpha matte :math:`\alpha` the foreground pixels and background pixels can be estimated. We refer to this process as foreground estimation.

To estimate the alpha matte Pymatting implements the following methods:

* Closed-form matting :cite:`levin2007closed`
* KNN matting :cite:`chen2013knn`
* Large kernel matting :cite:`he2010fast`
* Learning-based matting :cite:`zheng2009learning`
* Random-walk matting :cite:`grady2005random`
* Shared matting :cite:`GastalOliveira2010SharedMatting`

## Foreground Extraction

Simply multiplying the alpha matte with the input image results in halo artifacts. This motivates the development of foreground extraction methods.

.. html::

    <div class="figure">
        <img src="/figures/lemur_color_bleeding.png">
        <div class="caption">Figure 2: Input image naively composed onto a grey background (left) and extracted foreground placed onto the same background (right).</div>
    </div>

The following foreground estimation methods are implemented in PyMatting:

* Closed-form foreground estimation :cite:`levin2007closed`
* Fast multi-level foreground estimation :cite:`germer2020multilevel`

## Thanks

We thank Mathias Appel for [his lemur photo](https://www.flickr.com/photos/mathiasappel/25419442300/), licensed under [CC0 1.0 Universal (CC0 1.0) Public Domain License](https://creativecommons.org/publicdomain/zero/1.0/).
