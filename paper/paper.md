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
    orcid: 0000-0003-0872-7098
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

# Summary

An important step of many image editing tasks is to extract specific objects
from an image in order to place them in a scene of a movie or compose them onto
another background. Alpha matting describes the problem of separating the objects
in the foreground from the background of an image given only a rough sketch.
We introduce the PyMatting package for Python which implements various approaches
to solve the alpha matting problem. Our toolbox is also able to extract the
foreground of an image given the alpha matte. The implementation aims to be
computationally efficient and easy to use. 


# References