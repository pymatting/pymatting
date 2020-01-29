# PyMatting: A Python Library for Alpha Matting
[![License: MIT](https://img.shields.io/github/license/tuelwer/phase-retrieval)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/pymatting/pymatting.svg?branch=master)](https://travis-ci.org/pymatting/pymatting)

We introduce the PyMatting package for Python which implements various methods to solve the alpha matting problem.

- **Website and Documentation:** [https://pymatting.github.io/](https://pymatting.github.io)
- **Benchmarks:**  [https://pymatting.github.io/benchmark.html](https://pymatting.github.io/benchmark.html)

![Lemur](https://github.com/pymatting/pymatting/raw/master/data/lemur/lemur_at_the_beach.png)

Given an input image and a hand-drawn trimap (top row), alpha matting estimates the alpha channel of a foreground object which can then be composed onto a different background (bottom row).

PyMatting provides:
- Alpha matting implementations for:
  - Closed Form Alpha Matting [[1]](#1)
  - Large Kernel Matting [[2]](#2)
  - KNN Matting [[3]](#3)
  - Learning Based Digital Matting [[4]](#4)
  - Random Walk Matting [[5]](#5)
- Foreground estimation implementations for:
  - Closed Form Foreground Estimation [[1]](#1)
  - Multilevel Foreground Estimation (CPU, CUDA and OpenCL)
- Fast multithreaded KNN search
- Preconditioners to accelerate the convergence rate of conjugate gradient descent:
  - The *incomplete thresholded Cholesky decomposition* (*Incomplete* is part of the name. The implementation is quite complete.)
  - The V-Cycle Geometric Multigrid preconditioner
- Readable code leveraging [NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/scipylib/index.html) and [Numba](http://numba.pydata.org/)

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

### Installation
```
git clone https://github.com/pymatting/pymatting
cd pymatting
pip3 install .
```

## Very Short Example
```python
from pymatting import cutout

cutout(
    # input image path
    "data/lemur.png",
    # input trimap path
    "data/lemur_trimap.png",
    # output cutout path
    "lemur_cutout.png")
```

[More examples](https://github.com/pymatting/pymatting/tree/master/examples)

### Testing

Run the tests from the main directory:
```
 python3 tests/download_images.py
 pip3 install -r requirements_tests.txt
 pytest
```

Currently 89% of the code is covered by tests.

## Authors

- **Thomas Germer**
- **Tobias Uelwer**
- **Stefan Conrad**
- **Stefan Harmeling**

See also the list of [contributors](https://github.com/pymatting/pymatting/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

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
