---
title: 'PyMatting: A Python Library for Alpha Matting'
tags:
  - alpha matting
  - soft-segmentation
  - foreground extraction
  - toolbox
  - Python
authors:
  - name: Thomas Germer
    affiliation: 1
  - name: Tobias Uelwer
    affiliation: 1
  - name: Stefan Conrad
    affiliation: 1
  - name: Stefan Harmeling
    affiliation: 1
affiliations:
 - name: Department of Computer Science, Heinrich-Heine-Universtität Düsseldorf
   index: 1
date: 12 June 2020
bibliography: paper.bib
---

# Summary

An important step of many image editing tasks is to extract specific objects
from an image in order to place them in a scene of a movie or compose them onto
another background. Alpha matting describes the problem of separating the objects
in the foreground from the background of an image given only a rough sketch.
We introduce the PyMatting package for Python which implements various approaches
to solve the alpha matting problem. Our toolbox is also able to extract the
foreground of an image given the alpha matte. The implementation aims to be
computationally efficient and easy to use. 

For an image $I$ with foreground pixels $F$ and background pixels $B$,
alpha matting asks to determine opacities $\alpha$, such that the equality
\begin{equation}
I_i = \alpha_i F_i + (1-\alpha_i)B_i
\end{equation}
holds for every pixel $i$. This problem is ill-posed since,
for each pixel, we have three equations (one for each color channel) with
seven unknown variables. The implemented methods rely on a trimap, which is a
rough classification of the input image into foreground, background and unknown
pixels, to further constrain the problem. Subsequently, the foreground $F$ can be 
extracted from the input image $I$ and the previously computed alpha matte $\alpha$ 
using a foreground estimation method.


## Implemented Methods for Alpha Matting

- Closed-form Matting:
@levin2007closed show that assuming local smoothness of pixel colors yields a closed-form solution to the alpha matting problem. 

- KNN Matting:
@lee2011nonlocal and @chen2013knn use nearest neighbor information to derive closed-form solutions to the alpha matting problem which they note to perform particularly well on sparse trimaps.

- Large Kernel Matting:
@he2010fast propose an efficient algorithm based on a large kernel matting Laplacian.
They show that the computational complexity of their method is independent of the kernel size.

- Random Walk Matting:
@grady2005random use random walks on the pixels to estimate alpha. 
The calculated alpha of a pixel is the probability that a random walk starting from that pixel will reach a foreground pixel before encountering a background pixel.

- Learning Based Digital Matting:
@zheng2009learning estimate alpha using local semi-supervised learning. 
They assume that the alpha value of a pixel can be learned by a linear combination of the neighboring pixels.


## Implemented Methods for Foreground Estimation

- Closed-form Foreground Estimation:
For given $\alpha$, the foreground pixels $F$ can be determined by making additional smoothness assumptions on $F$ and $B$. 
Our toolbox implements the foreground estimation by @levin2007closed.

- Multi-level Foreground Estimation:
Furthermore, the PyMatting toolbox implements a novel multi-level approach for foreground estimation.
For this method our toolbox also provides GPU implementations for OpenCL and CUDA.

![Input image (top left) and input trimap (top right) are used to estimate an alpha matte (bottom left) and a foreground (bottom right) using the Pymatting library. Input image and input trimap are taken from @rhemann2009perceptually. 
\label{fig:grid}](figures/image_grid.png)

## Installation and Code Example

The following code snippet demonstrates the usage of the library:

```python
from pymatting import *
image = load_image("plant_image.png", "RGB")
trimap = load_image("plant_trimap.png", "GRAY")
alpha = estimate_alpha_cf(image, trimap)
foreground = estimate_foreground_cf(image, alpha)
cutout = stack_images(foreground, alpha)
save_image("results.png", cutout)
```

The $\texttt{estimate\_alpha\_cf}$ method implements closed-form alpha estimation, whereas the $\texttt{estimate\_foreground\_cf}$ method implements the closed-form foreground estimation [@levin2007closed]. 
The $\texttt{stack\_images}$ method can be used to compose the foreground onto a new background.

More code examples at different levels of abstraction can be found in the documentation of the toolbox. PyMatting can be easily installed through pip.


## Performance Comparison

Since all of the considered methods require to solve a large sparse systems of linear equations, an efficient solver is crucial for good performance. 
Therefore, the PyMatting package implements the conjugate gradients method [@hestenes1952methods] together with different preconditioners that improve convergence:
Jacobi, V-cycle [@lee2014scalable] and thresholded incomplete Cholesky decomposition [@jones1995improved].

To evaluate the performance of our implementation, we calculate the mean squared error on the unknown pixels of the benchmark images of @rhemann2009perceptually. 
\autoref{fig:errors} shows the  mean squared error to the ground truth alpha matte.
Our results are consistent with the results achieved by the authors implementation (if available).

![Mean squared error of the estimated alpha matte to the ground truth alpha matte.\label{fig:errors}](figures/laplacian_quality_many_bars.pdf)

![Comparison of peak memory usage in MB (left) and computational time (right) of our implementation of the preconditioned CG method with other solvers for closed-form matting.\label{fig:memory-runtime}](figures/memory_usage_and_running_time-crop.pdf)

![Comparison of runtime for different image sizes.\label{fig:runtimes}](figures/time_image_size-crop.pdf)

We compare the computational runtime of our solver with other solvers: PyAMG [@pyamg], UMFPACK [@umfpack], AMGCL [@amgcl], MUMPS [@MUMPS-a, MUMPS-b], Eigen [@eigen] and SuperLU [@li1999superlu]. \autoref{fig:runtimes} shows that our implemented conjugate gradients method in combination with the incomplete Cholesky decomposition preconditioner outperforms the other methods by a large margin. For the iterative solver, we use an absolute tolerance of $10^{-7}$, which we scale with the number of known pixels, i.e., pixels that are either marked as foreground or background in the trimap. The benchmarked linear system arises from the matting Laplacian by @levin2007closed. \autoref{fig:memory-runtime} shows that our solver also outperforms the other solvers in terms of memory usage.


## Compatibility and Extendability

The PyMatting package has been tested on Windows 10, Ubuntu 16.04 and macOS 10.15.2.
The package can be easily extended by adding new definitions of the graph Laplacian matrix. 
We plan on continuously extending our toolbox with new methods.

# References
