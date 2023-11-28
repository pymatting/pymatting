import numpy as np
import cupy as cp
from pymatting.util.util import div_round_up

_resize_nearest = cp.RawKernel(
    r"""
extern "C" __global__
void resize_nearest(
    float *dst,
    const float *src,
    int w_src, int h_src,
    int w_dst, int h_dst,
    int depth
){
    int x_dst = blockDim.x * blockIdx.x + threadIdx.x;
    int y_dst = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_dst >= w_dst || y_dst >= h_dst) return;

    int x_src = min(x_dst * w_src / w_dst, w_src - 1);
    int y_src = min(y_dst * h_src / h_dst, h_src - 1);

          float *ptr_dst = dst + (x_dst + y_dst * w_dst) * depth;
    const float *ptr_src = src + (x_src + y_src * w_src) * depth;

    for (int channel = 0; channel < depth; channel++){
        ptr_dst[channel] = ptr_src[channel];
    }
}
""",
    "resize_nearest",
)

ml_iteration = cp.RawKernel(
    r"""
extern "C" __global__
void ml_iteration(
          float *F,
          float *B,
    const float *F_prev,
    const float *B_prev,
    const float *image,
    const float *alpha,
    int w,
    int h,
    float regularization
){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int i = x + y * w;

    if (x >= w || y >= h) return;

    float a0 = alpha[i];
    float a1 = 1.0f - a0;

    float b00 = a0 * image[i * 3 + 0];
    float b01 = a0 * image[i * 3 + 1];
    float b02 = a0 * image[i * 3 + 2];

    float b10 = a1 * image[i * 3 + 0];
    float b11 = a1 * image[i * 3 + 1];
    float b12 = a1 * image[i * 3 + 2];

    int js[4] = {
        max(    0, x - 1) + y * w,
        min(w - 1, x + 1) + y * w,
        x + max(    0, y - 1) * w,
        x + min(h - 1, y + 1) * w,
    };

    float a_sum = 0.0f;

    for (int d = 0; d < 4; d++){
        int j = js[d];

        float da = regularization + fabsf(a0 - alpha[j]);

        a_sum += da;

        b00 += da * F_prev[j * 3 + 0];
        b01 += da * F_prev[j * 3 + 1];
        b02 += da * F_prev[j * 3 + 2];

        b10 += da * B_prev[j * 3 + 0];
        b11 += da * B_prev[j * 3 + 1];
        b12 += da * B_prev[j * 3 + 2];
    }

    float a00 = a0 * a0 + a_sum;
    float a11 = a1 * a1 + a_sum;
    float a01 = a0 * a1;

    float inv_det = 1.0f / (a00 * a11 - a01 * a01);

    F[i * 3 + 0] = fmaxf(0.0f, fminf(1.0f, inv_det * (a11 * b00 - a01 * b10)));
    F[i * 3 + 1] = fmaxf(0.0f, fminf(1.0f, inv_det * (a11 * b01 - a01 * b11)));
    F[i * 3 + 2] = fmaxf(0.0f, fminf(1.0f, inv_det * (a11 * b02 - a01 * b12)));

    B[i * 3 + 0] = fmaxf(0.0f, fminf(1.0f, inv_det * (a00 * b10 - a01 * b00)));
    B[i * 3 + 1] = fmaxf(0.0f, fminf(1.0f, inv_det * (a00 * b11 - a01 * b01)));
    B[i * 3 + 2] = fmaxf(0.0f, fminf(1.0f, inv_det * (a00 * b12 - a01 * b02)));
}
""",
    "ml_iteration",
)


def estimate_foreground_ml_cupy(
    input_image,
    input_alpha,
    regularization=1e-5,
    n_small_iterations=10,
    n_big_iterations=2,
    small_size=32,
    block_size=(32, 32),
    return_background=False,
    to_numpy=True,
):
    """See the :code:`estimate_foreground` method for documentation."""
    h0, w0, depth = input_image.shape

    assert depth == 3

    n = h0 * w0 * depth

    if isinstance(input_image, cp.ndarray):
        input_image = input_image.astype(cp.float32).ravel()
    else:
        input_image = cp.asarray(input_image.astype(np.float32).flatten())

    if isinstance(input_alpha, cp.ndarray):
        input_alpha = input_alpha.astype(cp.float32).ravel()
    else:
        input_alpha = cp.asarray(input_alpha.astype(np.float32).flatten())

    F_prev = cp.zeros(n, dtype=cp.float32)
    B_prev = cp.zeros(n, dtype=cp.float32)

    F = cp.zeros(n, dtype=cp.float32)
    B = cp.zeros(n, dtype=cp.float32)

    image_level = cp.zeros(n, dtype=cp.float32)
    alpha_level = cp.zeros(h0 * w0, dtype=cp.float32)

    n_levels = (max(w0, h0) - 1).bit_length()

    w_prev = 1
    h_prev = 1

    def resize_nearest(dst, src, w_src, h_src, w_dst, h_dst, depth):
        grid_size = (
            div_round_up(w_dst, block_size[0]),
            div_round_up(h_dst, block_size[1]),
        )

        _resize_nearest(
            grid_size,
            block_size,
            (
                dst,
                src,
                w_src,
                h_src,
                w_dst,
                h_dst,
                depth,
            ),
        )

    resize_nearest(F, input_image, w0, h0, w_prev, h_prev, depth)
    resize_nearest(B, input_image, w0, h0, w_prev, h_prev, depth)

    for i_level in range(n_levels + 1):
        w = round(w0 ** (i_level / n_levels))
        h = round(h0 ** (i_level / n_levels))

        resize_nearest(image_level, input_image, w0, h0, w, h, depth)
        resize_nearest(alpha_level, input_alpha, w0, h0, w, h, 1)

        resize_nearest(F_prev, F, w_prev, h_prev, w, h, depth)
        resize_nearest(B_prev, B, w_prev, h_prev, w, h, depth)

        # Do more iterations for low resolution.
        n_iter = n_big_iterations
        if min(w, h) <= small_size:
            n_iter = n_small_iterations

        grid_size = (div_round_up(w, block_size[0]), div_round_up(h, block_size[1]))

        for i_iter in range(n_iter):
            ml_iteration(
                grid_size,
                block_size,
                (
                    F,
                    B,
                    F_prev,
                    B_prev,
                    image_level,
                    alpha_level,
                    w,
                    h,
                    np.float32(regularization),
                ),
            )

            F_prev, F = F, F_prev
            B_prev, B = B, B_prev

        w_prev = w
        h_prev = h

    # Reshape back to original shape.
    F = F.reshape(h0, w0, depth)
    B = B.reshape(h0, w0, depth)

    # Convert to NumPy if requested
    if to_numpy:
        F_out = cp.asnumpy(F)
        B_out = cp.asnumpy(B)
    else:
        F_out = F
        B_out = B

    if return_background:
        return F_out, B_out

    return F_out
