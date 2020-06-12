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
pixels, to further constrain the problem. Giving the alpha matte, foreground estimation
aims to extract the foreground $F$ from image $I$.

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

# References