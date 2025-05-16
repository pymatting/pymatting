import math
import numpy as np
import pyopencl as cl

source = """
__kernel void resize_nearest(
    __global float *dst,
    __global const float *src,
    int w_src, int h_src,
    int w_dst, int h_dst,
    int depth
){
    int x_dst = get_global_id(0);
    int y_dst = get_global_id(1);

    if (x_dst >= w_dst || y_dst >= h_dst) return;

    int x_src = min(x_dst * w_src / w_dst, w_src - 1);
    int y_src = min(y_dst * h_src / h_dst, h_src - 1);

    __global       float *ptr_dst = dst + (x_dst + y_dst * w_dst) * depth;
    __global const float *ptr_src = src + (x_src + y_src * w_src) * depth;

    for (int channel = 0; channel < depth; channel++){
        ptr_dst[channel] = ptr_src[channel];
    }
}

__kernel void ml_iteration(
    __global       float *F,
    __global       float *B,
    __global const float *F_prev,
    __global const float *B_prev,
    __global const float *image,
    __global const float *alpha,
    int w,
    int h,
    float regularization
){
    int x = get_global_id(0);
    int y = get_global_id(1);

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

        float da = regularization + fabs(a0 - alpha[j]);

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

    F[i * 3 + 0] = fmax(0.0f, fmin(1.0f, inv_det * (a11 * b00 - a01 * b10)));
    F[i * 3 + 1] = fmax(0.0f, fmin(1.0f, inv_det * (a11 * b01 - a01 * b11)));
    F[i * 3 + 2] = fmax(0.0f, fmin(1.0f, inv_det * (a11 * b02 - a01 * b12)));
    B[i * 3 + 0] = fmax(0.0f, fmin(1.0f, inv_det * (a00 * b10 - a01 * b00)));
    B[i * 3 + 1] = fmax(0.0f, fmin(1.0f, inv_det * (a00 * b11 - a01 * b01)));
    B[i * 3 + 2] = fmax(0.0f, fmin(1.0f, inv_det * (a00 * b12 - a01 * b02)));
}
"""

platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.GPU)[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
program = cl.Program(context, source).build()


def estimate_foreground_ml_pyopencl(
    input_image,
    input_alpha,
    regularization=1e-5,
    n_small_iterations=10,
    n_big_iterations=2,
    small_size=32,
    return_background=False,
):
    """See the :code:`estimate_foreground` method for documentation."""

    def upload(array):
        hostbuf = array.astype(np.float32).flatten()
        return cl.Buffer(
            context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=hostbuf,
        )

    def alloc(*shape):
        n = math.prod(shape)
        buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, n * 4)
        zero = np.uint32(0)
        cl.enqueue_fill_buffer(queue, buffer, zero, 0, n*4)
        return buffer

    def download(device_buf, shape):
        host_buf = np.empty(shape, dtype=np.float32)

        cl.enqueue_copy(queue, host_buf, device_buf)

        return host_buf.reshape(shape)

    h0, w0, depth = input_image.shape

    assert depth == 3

    n = h0 * w0 * depth

    input_image = upload(input_image)
    input_alpha = upload(input_alpha)

    F_prev = alloc(n)
    B_prev = alloc(n)

    F = alloc(n)
    B = alloc(n)

    image_level = alloc(n)
    alpha_level = alloc(h0 * w0)

    n_levels = (max(w0, h0) - 1).bit_length()

    def resize_nearest(dst, src, w_src, h_src, w_dst, h_dst, depth):
        program.resize_nearest(
            queue,
            (w_dst, h_dst),
            None,
            dst,
            src,
            *np.int32([w_src, h_src, w_dst, h_dst, depth])
        )

    w_prev = 1
    h_prev = 1

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

        for i_iter in range(n_iter):
            program.ml_iteration(
                queue,
                (w, h),
                None,
                F,
                B,
                F_prev,
                B_prev,
                image_level,
                alpha_level,
                np.int32(w),
                np.int32(h),
                np.float32(regularization),
            )

            F_prev, F = F, F_prev
            B_prev, B = B, B_prev

        w_prev = w
        h_prev = h

    F_host = download(F, (h0, w0, depth))
    B_host = download(B, (h0, w0, depth))

    for buf in [
        F,
        B,
        F_prev,
        B_prev,
        input_image,
        input_alpha,
        image_level,
        alpha_level,
    ]:
        buf.release()

    if return_background:
        return F_host, B_host

    return F_host
