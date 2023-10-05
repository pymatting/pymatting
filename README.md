# PyMatting: A Python Library for Alpha Matting
[![License: MIT](https://img.shields.io/github/license/pymatting/pymatting?color=brightgreen)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/pymatting/pymatting/.github/workflows/tests.yml?branch=master)](https://github.com/pymatting/pymatting/actions?query=workflow%3Atests)
[![PyPI](https://img.shields.io/pypi/v/pymatting)](https://pypi.org/project/PyMatting/)
[![JOSS](https://joss.theoj.org/papers/9766cab65bfbf07a70c8a835edd3875a/status.svg)](https://joss.theoj.org/papers/9766cab65bfbf07a70c8a835edd3875a)
[![Gitter](https://img.shields.io/gitter/room/pymatting/pymatting)](https://gitter.im/pymatting/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

We introduce the PyMatting package for Python which implements various methods to solve the alpha matting problem.

- **Website and Documentation:** [https://pymatting.github.io/](https://pymatting.github.io)
- **Benchmarks:**  [https://pymatting.github.io/benchmarks.html](https://pymatting.github.io/benchmarks.html)

![Lemur](https://github.com/pymatting/pymatting/raw/master/data/lemur/lemur_at_the_beach.png)

Given an input image and a hand-drawn trimap (top row), alpha matting estimates the alpha channel of a foreground object which can then be composed onto a different background (bottom row).

PyMatting provides:
- Alpha matting implementations for:
  - Closed Form Alpha Matting [[1]](#1)
  - Large Kernel Matting [[2]](#2)
  - KNN Matting [[3]](#3)
  - Learning Based Digital Matting [[4]](#4)
  - Random Walk Matting [[5]](#5)
  - Shared Sampling Matting [[6]](#6)
- Foreground estimation implementations for:
  - Closed Form Foreground Estimation [[1]](#1)
  - Fast Multi-Level Foreground Estimation (CPU, CUDA and OpenCL) [[7]](#7)
- Fast multithreaded KNN search
- Preconditioners to accelerate the convergence rate of conjugate gradient descent:
  - The *incomplete thresholded Cholesky decomposition* (*Incomplete* is part of the name. The implementation is quite complete.)
  - The V-Cycle Geometric Multigrid preconditioner
- Readable code leveraging [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [Numba](http://numba.pydata.org/)

## Getting Started

### Requirements

Minimal requiremens
* numpy>=1.16.0
* pillow>=5.2.0
* numba>=0.47.0
* scipy>=1.1.0

Additional requirements for GPU support
* cupy-cuda90>=6.5.0 or similar
* pyopencl>=2019.1.2

Requirements to run the tests
* pytest>=5.3.4

### Installation with PyPI

```bash
pip3 install pymatting
```

### Installation from Source

```bash
git clone https://github.com/pymatting/pymatting
cd pymatting
pip3 install .
```

## Example
```python
from pymatting import cutout

cutout(
    # input image path
    "data/lemur/lemur.png",
    # input trimap path
    "data/lemur/lemur_trimap.png",
    # output cutout path
    "lemur_cutout.png")
```

[More advanced examples](https://pymatting.github.io/examples.html)

## Trimap Construction

All implemented methods rely on trimaps which roughly classify the image into foreground, background and unknown reagions.
Trimaps are expected to be `numpy.ndarrays` of type `np.float64`  having the same shape as the input image with only one color-channel.
Trimap values of 0.0 denote pixels which are 100% background.
Similarly, trimap values of 1.0 denote pixels which are 100% foreground.
All other values indicate unknown pixels which will be estimated by the algorithm.


## Testing

Run the tests from the main directory:
```
 python3 tests/download_images.py
 pip3 install -r requirements_tests.txt
 pytest
```

Currently 89% of the code is covered by tests.

## Upgrade

```bash
pip3 install --upgrade pymatting
python3 -c "import pymatting"
```

## Bug Reports, Questions and Pull-Requests

Please, see [our community guidelines](https://github.com/pymatting/pymatting/blob/master/CONTRIBUTING.md).

## Authors

- **Thomas Germer**
- **Tobias Uelwer**
- **Stefan Conrad**
- **Stefan Harmeling**

See also the list of [contributors](https://github.com/pymatting/pymatting/contributors) who participated in this project.

## Projects using PyMatting

* [Rembg](https://github.com/danielgatis/rembg) - an excellent tool for removing image backgrounds.
* [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) - a library for a wide range of image segmentation tasks.
* [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) - a node-based image processing GUI.
* [LSA-Matting](https://github.com/kfeng123/LSA-Matting) - improving deep image matting via local smoothness assumption.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citing

If you found PyMatting to be useful for your work, please consider citing our [paper](https://doi.org/10.21105/joss.02481):

```
@article{Germer2020,
  doi = {10.21105/joss.02481},
  url = {https://doi.org/10.21105/joss.02481},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {54},
  pages = {2481},
  author = {Thomas Germer and Tobias Uelwer and Stefan Conrad and Stefan Harmeling},
  title = {PyMatting: A Python Library for Alpha Matting},
  journal = {Journal of Open Source Software}
}
```

## References

<a id="1">[1]</a>
Anat Levin, Dani Lischinski, and Yair Weiss. A closed-form solution to natural image matting. IEEE transactions on pattern analysis and machine intelligence, 30(2):228–242, 2007.

<a id="2">[2]</a>
Kaiming He, Jian Sun, and Xiaoou Tang. Fast matting using large kernel matting laplacian matrices. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2165–2172. IEEE, 2010.

<a id="3">[3]</a>
Qifeng Chen, Dingzeyu Li, and Chi-Keung Tang. Knn matting. IEEE transactions on pattern analysis and machine intelligence, 35(9):2175–2188, 2013.

<a id="4">[4]</a>
Yuanjie Zheng and Chandra Kambhamettu. Learning based digital matting. In 2009 IEEE 12th international conference on computer vision, 889–896. IEEE, 2009.

<a id="5">[5]</a>
Leo Grady, Thomas Schiwietz, Shmuel Aharon, and Rüdiger Westermann. Random walks for interactive alpha-matting. In Proceedings of VIIP, volume 2005, 423–429. 2005.

<a id="6">[6]</a>
Eduardo S. L. Gastal and Manuel M. Oliveira. "Shared Sampling for Real-Time Alpha Matting". Computer Graphics Forum. Volume 29 (2010), Number 2, Proceedings of Eurographics 2010, pp. 575-584.

<a id="7">[7]</a>
Germer, T., Uelwer, T., Conrad, S., & Harmeling, S. (2020). Fast Multi-Level Foreground Estimation. arXiv preprint arXiv:2006.14970.

Lemur image by Mathias Appel from https://www.flickr.com/photos/mathiasappel/25419442300/ licensed under [CC0 1.0 Universal (CC0 1.0) Public Domain License](https://creativecommons.org/publicdomain/zero/1.0/).
