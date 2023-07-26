## Quality

To evaluate the performance of our implementation we calculate the mean squared error on the unknown pixels of the benchmark images of :cite:`rhemann2009perceptually`.

.. html::
    <div class="figure">
        <img src="/figures/laplacian_quality_many_bars.png">
        <div class="caption">Figure 1: Mean squared error of the estimated alpha matte to the ground truth alpha matte.</div>
    </div>

.. html::
    <div class="figure">
        <img src="/figures/laplacians.png">
        <div class="caption">Figure 2: Mean squared error across all images from the benchmark dataset.</div>
    </div>

## Visualization

The following videos show the iterates of the different methods.
Note that the videos of the slower methods have been accelerated.

.. html::
    <table style="width:100%;display:inline-block;">
        <tr align="center">
        <td>
            <embed>
                <video width="320" height="180" loop autoplay muted playsinline>
                    <source src="https://github.com/pymatting/videos/blob/master/cf_web.mp4?raw=true" type="video/mp4">
                </video>
            </embed>
        </td>
        <td>
            <embed>
                <video width="320" height="180" loop autoplay muted playsinline>
                    <source src="https://github.com/pymatting/videos/blob/master/knn_web.mp4?raw=true" type="video/mp4">
                </video>
            </embed>
        </td>
    </tr>
    <tr align="center">
        <td>CF</td>
        <td>KNN</td>
    </tr>
    <tr align="center">
        <td>
            <embed>
                <video width="320" height="180" loop autoplay muted playsinline>
                    <source src="https://github.com/pymatting/videos/blob/master/lkm_web.mp4?raw=true" type="video/mp4">
                </video>
            </embed>
        </td>
        <td>
            <embed>
                <video width="320" height="180" loop autoplay muted playsinline>
                    <source src="https://github.com/pymatting/videos/blob/master/rw_web.mp4?raw=true" type="video/mp4">
                </video>
            </embed>
        </td>
        </tr>
        <tr align="center">
            <td>LKM</td>
            <td>RW</td>
        </tr>
    </table>

## Performance

We compare the computational runtime of our solver with other solvers: pyAMG, UMFPAC, AMGCL, MUMPS, Eigen and SuperLU. Figure 3 shows that our implemented conjugate gradients method in combination with the incomplete Cholesky decomposition preconditioner outperforms the other methods by a large margin. For the iterative solver we used an absolute tolerance of :math:`10^{-7}`, which we scaled with the number of known pixels, i.e. pixels that are either marked as foreground or background in the trimap.


.. html::
    <div class="figure">
        <img src="/figures/time_image_size.png">
        <div class="caption">Figure 3: Comparison of runtime for different image sizes.</div>
    </div>

.. html::
    <div class="figure">
        <img src="/figures/average_running_time.png">
        <div class="caption">Figure 4: Mean running time of each solver in seconds.</div>
    </div>

.. html::
    <div class="figure">
        <img src="/figures/average_peak_memory_usage.png">
        <div class="caption">Figure 5: Peak memory for each solver usage in MB.</div>
    </div>
