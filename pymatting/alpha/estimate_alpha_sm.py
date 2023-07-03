from numba import njit, pndindex
import numpy as np

def estimate_alpha_sm(
    image,
    trimap,
    return_foreground_background=False,
    trimap_expansion_radius=10,
    trimap_expansion_threshold=0.02,
    sample_gathering_angles=4,
    sample_gathering_weights=(3.0, 2.0, 1.0, 4.0),
    sample_gathering_Np_radius=1,
    sample_refinement_radius=5,
    local_smoothing_radius1=5,
    local_smoothing_radius2=5,
    local_smoothing_radius3=5,
    local_smoothing_sigma_sq1=None,
    local_smoothing_sigma_sq2=0.1,
    local_smoothing_sigma_sq3=None,

):
    """
    Estimate alpha from an input image and an input trimap using Shared Matting as proposed by :cite:`GastalOliveira2010SharedMatting`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h \\times  w \\times d` for which the alpha matte should be estimated
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times  w` of the image
    return_foreground_background: numpy.ndarray
        Whether to return foreground and background estimate. They will be computed either way
    trimap_expansion_radius: int
        How much to expand trimap.
    trimap_expansion_threshold: float
        Which pixel colors are similar enough to expand trimap into
    sample_gathering_angles: int
        In how many directions to search for new samples.
    sample_gathering_weights: Tuple[float, float, float, float]
        Weights for various cost functions
    sample_gathering_Np_radius: int
        Radius of Np function
    sample_refinement_radius: int
        Search region for better neighboring samples
    local_smoothing_radius1: int
        Radius for foreground/background smoothing
    local_smoothing_radius2: int
        Radius for confidence computation
    local_smoothing_radius3: int
        Radius for low frequency alpha computation
    local_smoothing_sigma_sq1: float
        Squared sigma value for foreground/background smoothing
        Defaults to :code:`(2 * local_smoothing_radius1 + 1)**2 / (9 * pi)` if not given
    local_smoothing_sigma_sq2: float
        Squared sigma value for confidence computation
    local_smoothing_sigma_sq3: float
        Squared sigma value for low frequency alpha computation
        Defaults to :code:`(2 * local_smoothing_radius3 + 1)**2 / (9 * pi)` if not given

    Returns
    -------
    alpha: numpy.ndarray
        Estimated alpha matte
    foreground: numpy.ndarray
        Estimated foreground
    background: numpy.ndarray
        Estimated background

    Example
    -------
    >>> from pymatting import *
    >>> image = load_image("data/lemur/lemur.png", "RGB")
    >>> trimap = load_image("data/lemur/lemur_trimap.png", "GRAY")
    >>> alpha, foreground, background = estimate_alpha_sm(
    ...     image,
    ...     trimap,
    ...     return_foreground_background=True,
    ...     sample_gathering_angles=4)
    """
    assert image.dtype in [np.float32, np.float64], f"image.dtype should be float32 or float64, but is {image.dtype}"
    assert trimap.dtype in [np.float32, np.float64], f"trimap.dtype should be float32 or float64, but is {trimap.dtype}"
    assert trimap.shape == image.shape[:2], f"image height and width should match trimap height and width, but image shape is {image.shape} and trimap shape is {trimap.shape}"
    assert len(image.shape) == 3 and image.shape[2] == 3, f"image should be RGB, but shape is {image.shape}"

    if local_smoothing_sigma_sq1 is None:
        local_smoothing_sigma_sq1 = (2 * local_smoothing_radius1 + 1)**2 / (9.0 * np.pi)

    if local_smoothing_sigma_sq3 is None:
        local_smoothing_sigma_sq3 = (2 * local_smoothing_radius1 + 1)**2 / (9.0 * np.pi)

    # Convert to float32 if float64. Precision should be sufficient.
    image = image.astype(np.float32)
    trimap = trimap.astype(np.float32)

    expanded_trimap = trimap.copy()

    if trimap_expansion_radius > 0 and trimap_expansion_threshold > 0.0:
        expand_trimap(
            expanded_trimap,
            trimap,
            image,
            k_i=trimap_expansion_radius,
            k_c=trimap_expansion_threshold)

    avg_F = image[expanded_trimap == 1.0].mean(axis=0)
    avg_B = image[expanded_trimap == 0.0].mean(axis=0)

    gathering_F = np.zeros_like(image) + avg_F
    gathering_B = np.zeros_like(image) + avg_B

    gathering_alpha = np.zeros_like(trimap)

    eN, eA, ef, eb = sample_gathering_weights

    sample_gathering(
        gathering_F,
        gathering_B,
        gathering_alpha,
        image,
        expanded_trimap,
        num_angles=sample_gathering_angles,
        eN=eN,
        eA=eA,
        ef=ef,
        eb=eb,
        Np_radius=sample_gathering_Np_radius)

    refined_F = np.zeros_like(image)
    refined_B = np.zeros_like(image)
    refined_alpha = np.zeros_like(trimap)

    sample_refinement(
        refined_F,
        refined_B,
        refined_alpha,
        gathering_F,
        gathering_B,
        image,
        expanded_trimap,
        radius=sample_refinement_radius)

    final_F = np.zeros_like(image)
    final_B = np.zeros_like(image)
    final_alpha = np.zeros_like(trimap)

    local_smoothing(
        final_F,
        final_B,
        final_alpha,
        refined_F,
        refined_B,
        refined_alpha,
        image,
        expanded_trimap,
        radius1=local_smoothing_radius1,
        radius2=local_smoothing_radius2,
        radius3=local_smoothing_radius3,
        sigma_sq1=local_smoothing_sigma_sq1,
        sigma_sq2=local_smoothing_sigma_sq2,
        sigma_sq3=local_smoothing_sigma_sq3)

    if return_foreground_background:
        return final_alpha, final_F, final_B
    else:
        return final_alpha

@njit("f4(f4[::1], f4[::1], f4[::1])", cache=True, nogil=True)
def estimate_alpha(I, F, B):
    fb0 = F[0] - B[0]
    fb1 = F[1] - B[1]
    fb2 = F[2] - B[2]

    ib0 = I[0] - B[0]
    ib1 = I[1] - B[1]
    ib2 = I[2] - B[2]

    denom = fb0 * fb0 + fb1 * fb1 + fb2 * fb2 + 1e-5

    alpha = (ib0 * fb0 + ib1 * fb1 + ib2 * fb2) / denom

    alpha = max(0.0, min(1.0, alpha))

    return alpha

@njit("f4(f4[::1], f4[::1])", cache=True, nogil=True)
def inner(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

@njit("f4(f4[::1], f4[::1], f4[::1])", cache=True, nogil=True)
def Mp2(I, F, B):
    a = estimate_alpha(I, F, B)

    d0 = a * F[0] + (1.0 - a) * B[0] - I[0]
    d1 = a * F[1] + (1.0 - a) * B[1] - I[1]
    d2 = a * F[2] + (1.0 - a) * B[2] - I[2]

    return d0 * d0 + d1 * d1 + d2 * d2

@njit("f4(f4[:, :, ::1], i8, i8, f4[::1], f4[::1], i8)", cache=True, nogil=True)
def Np(image, x, y, F, B, r):
    h, w = image.shape[:2]
    result = 0.0
    for y2 in range(y - r, y + r + 1):
        y2 = max(0, min(h - 1, y2))
        for x2 in range(x - r, x + r + 1):
            x2 = max(0, min(w - 1, x2))
            result += Mp2(image[y2, x2], F, B)
    return result

@njit("f4(f4[:, :, ::1], i8, i8, i8, i8)", cache=True, nogil=True)
def Ep(image, px, py, sx, sy):
    result = 0.0

    spx = sx - px
    spy = sy - py

    d = np.hypot(spx, spy)

    if d == 0.0: return 0.0

    num_steps = int(np.ceil(d))
    num_steps = max(1, min(10, num_steps))

    step_x = spx / num_steps
    step_y = spy / num_steps

    h, w = image.shape[:2]

    for i in range(num_steps + 1):
        qx = int(px + float(i) * step_x)
        qy = int(py + float(i) * step_x)

        q_l = max(0, min(w - 1, qx - 1))
        q_r = max(0, min(w - 1, qx + 1))
        q_u = max(0, min(h - 1, qy + 1))
        q_d = max(0, min(h - 1, qy - 1))
        qx = max(0, min(w - 1, qx))
        qy = max(0, min(h - 1, qy))

        Ix0 = 0.5 * (image[qy, q_r, 0] - image[qy, q_l, 0])
        Ix1 = 0.5 * (image[qy, q_r, 1] - image[qy, q_l, 1])
        Ix2 = 0.5 * (image[qy, q_r, 2] - image[qy, q_l, 2])

        Iy0 = 0.5 * (image[q_u, qx, 0] - image[q_d, qx, 0])
        Iy1 = 0.5 * (image[q_u, qx, 1] - image[q_d, qx, 1])
        Iy2 = 0.5 * (image[q_u, qx, 2] - image[q_d, qx, 2])

        v0 = step_x * Ix0 + step_y * Iy0
        v1 = step_x * Ix1 + step_y * Iy1
        v2 = step_x * Ix2 + step_y * Iy2

        result += np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)

    return result

@njit("f4(f4[::1], f4[::1])", cache=True, nogil=True)
def dist(a, b):
    d2 = 0.0
    for i in range(a.shape[0]):
        d2 += (a[i] - b[i]) ** 2
    return np.sqrt(d2)

@njit("f4(f4[::1])", cache=True, nogil=True)
def length(a):
    return np.sqrt(inner(a, a))

@njit("void(f4[:, ::1], f4[:, ::1], f4[:, :, ::1], i8, f4)", cache=True, parallel=True, nogil=True)
def expand_trimap(expanded_trimap, trimap, image, k_i, k_c):
    # NB: Description in paper does not match published test images.
    # The radius appears to be larger and  expanded trimap is sparser.
    h, w = trimap.shape

    for y, x in pndindex((h, w)):
        if trimap[y, x] == 0 or trimap[y, x] == 1: continue

        closest = np.inf

        for y2 in range(y - k_i, y + k_i + 1):
            for x2 in range(x - k_i, x + k_i + 1):
                if x2 < 0 or x2 >= w or y2 < 0 or y2 >= h: continue
                if trimap[y2, x2] != 0 and trimap[y2, x2] != 1: continue

                dr = image[y, x, 0] - image[y2, x2, 0]
                dg = image[y, x, 1] - image[y2, x2, 1]
                db = image[y, x, 2] - image[y2, x2, 2]

                color_distance = np.sqrt(dr * dr + dg * dg + db * db)

                spatial_distance = np.hypot(x - x2, y - y2)

                if color_distance > k_c: continue
                if spatial_distance > k_i: continue

                if spatial_distance < closest:
                    closest = spatial_distance
                    expanded_trimap[y, x] = trimap[y2, x2]

@njit("void(f4[:, :, ::1], f4[:, :, ::1], f4[:, ::1], f4[:, :, ::1], f4[:, ::1], i8, f4, f4, f4, f4, i8)", cache=True, parallel=True, nogil=True)
def sample_gathering(
    gathering_F,
    gathering_B,
    gathering_alpha,
    image,
    trimap,
    num_angles,
    eN,
    eA,
    ef,
    eb,
    Np_radius,
):
    h, w = trimap.shape

    max_steps = 2 * max(w, h)

    for y, x in pndindex((h, w)):
        fg_samples = np.zeros((num_angles, 3), dtype=np.float32)
        fg_samples_xy = np.zeros((num_angles, 2), dtype=np.int32)
        bg_samples = np.zeros((num_angles, 3), dtype=np.float32)
        bg_samples_xy = np.zeros((num_angles, 2), dtype=np.int32)

        C_p = image[y, x]

        gathering_alpha[y, x] = trimap[y, x]

        if trimap[y, x] == 0:
            gathering_B[y, x] = C_p
            continue

        if trimap[y, x] == 1:
            gathering_F[y, x] = C_p
            continue

        # Fixed start angles in 8-by-8 grid for reproducible tests
        n = 8
        i = (x % n) + (y % n) * n
        # Shuffle (99991 is a prime number)
        i = (i * 99991) % (n * n)
        start_angle = 2.0 * np.pi / (n * n) * i

        num_fg_samples = 0
        num_bg_samples = 0

        for i in range(num_angles):
            angle = 2.0 * np.pi / num_angles * i + start_angle

            c = np.cos(angle)
            s = np.sin(angle)

            has_fg = False
            has_bg = False

            for step in range(max_steps):
                if has_fg and has_bg: break

                x2 = int(x + step * c)
                y2 = int(y + step * s)

                if x2 < 0 or y2 < 0 or x2 >= w or y2 >= h: break

                if not has_fg and trimap[y2, x2] == 1:
                    fg_samples[num_fg_samples] = image[y2, x2]
                    fg_samples_xy[num_fg_samples, 0] = x2
                    fg_samples_xy[num_fg_samples, 1] = y2
                    num_fg_samples += 1
                    has_fg = True

                if not has_bg and trimap[y2, x2] == 0:
                    bg_samples[num_bg_samples] = image[y2, x2]
                    bg_samples_xy[num_bg_samples, 0] = x2
                    bg_samples_xy[num_bg_samples, 1] = y2
                    num_bg_samples += 1
                    has_bg = True

        if num_fg_samples == 0:
            fg_samples[num_fg_samples] = gathering_F[y, x]
            fg_samples_xy[num_fg_samples, 0] = x
            fg_samples_xy[num_fg_samples, 1] = y
            num_fg_samples += 1

        if num_bg_samples == 0:
            bg_samples[num_bg_samples] = gathering_B[y, x]
            bg_samples_xy[num_bg_samples, 0] = x
            bg_samples_xy[num_bg_samples, 1] = y
            num_bg_samples += 1

        min_Ep_f = np.inf
        min_Ep_b = np.inf

        for i in range(num_fg_samples):
            Ep_f = Ep(image, x, y, fg_samples_xy[i, 0], fg_samples_xy[i, 1])

            min_Ep_f = min(min_Ep_f, Ep_f)

        for j in range(num_bg_samples):
            Ep_b = Ep(image, x, y, bg_samples_xy[j, 0], bg_samples_xy[j, 1])

            min_Ep_b = min(min_Ep_b, Ep_b)

        PF_p = min_Ep_b / (min_Ep_f + min_Ep_b + 1e-5)

        min_cost = np.inf

        # Find best foreground/background pair
        for i in range(num_fg_samples):
            for j in range(num_bg_samples):
                F = fg_samples[i]
                B = bg_samples[j]

                alpha_p = estimate_alpha(C_p, F, B)

                Ap = PF_p + (1.0 - 2.0 * PF_p) * alpha_p

                Dp_f = np.hypot(x - fg_samples_xy[i, 0], y - fg_samples_xy[i, 1])
                Dp_b = np.hypot(x - bg_samples_xy[j, 0], y - bg_samples_xy[j, 1])

                g_p = (
                    Np(image, x, y, F, B, Np_radius)**eN *
                    Ap**eA *
                    Dp_f**ef *
                    Dp_b**eb)

                if min_cost > g_p:
                    min_cost = g_p
                    gathering_alpha[y, x] = alpha_p
                    gathering_F[y, x] = F
                    gathering_B[y, x] = B

@njit("void(f4[:, :, ::1], f4[:, :, ::1], f4[:, ::1], f4[:, :, ::1], f4[:, :, ::1], f4[:, :, ::1], f4[:, ::1], i8)", cache=True, parallel=False, nogil=True)
def sample_refinement(
    refined_F,
    refined_B,
    refined_alpha,
    gathering_F,
    gathering_B,
    image,
    trimap,
    radius,
):
    h, w = trimap.shape

    refined_F[:] = gathering_F
    refined_B[:] = gathering_B
    refined_alpha[:] = trimap

    for y, x in pndindex((h, w)):
        C_p = image[y, x]

        if trimap[y, x] == 0 or trimap[y, x] == 1:
            continue

        max_samples = 3
        sample_F = np.zeros((max_samples, 3), dtype=np.float32)
        sample_B = np.zeros((max_samples, 3), dtype=np.float32)
        sample_cost = np.zeros(max_samples, dtype=np.float32)
        sample_cost[:] = np.inf

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x2 = x + dx
                y2 = y + dy

                if 0 <= x2 < w and 0 <= y2 < h:
                    F_q = gathering_F[y2, x2]
                    B_q = gathering_B[y2, x2]

                    cost = Mp2(C_p, F_q, B_q)

                    i = np.argmax(sample_cost)

                    if cost < sample_cost[i]:
                        sample_cost[i] = cost
                        sample_F[i] = F_q
                        sample_B[i] = B_q

        F_mean = sample_F.sum(axis=0) / max_samples
        B_mean = sample_B.sum(axis=0) / max_samples

        refined_F[y, x] = F_mean
        refined_B[y, x] = B_mean

        refined_alpha[y, x] = estimate_alpha(C_p, F_mean, B_mean)

@njit("void(f4[:, :, ::1], f4[:, :, ::1], f4[:, ::1], f4[:, :, ::1], f4[:, :, ::1], f4[:, ::1], f4[:, :, ::1], f4[:, ::1], i8, i8, i8, f4, f4, f4)", cache=True, parallel=True, nogil=True)
def local_smoothing(
    final_F,
    final_B,
    final_alpha,
    refined_F,
    refined_B,
    refined_alpha,
    image,
    trimap,
    radius1,
    radius2,
    radius3,
    sigma_sq1,
    sigma_sq2,
    sigma_sq3,
):
    h, w = trimap.shape

    final_confidence = np.zeros((h, w), dtype=np.float32)
    W_FB = np.zeros((h, w), dtype=np.float32)
    low_frequency_alpha = np.zeros((h, w), dtype=np.float32)

    final_F[:] = refined_F
    final_B[:] = refined_B

    for y, x in pndindex((h, w)):
        C_p = image[y, x]

        if trimap[y, x] == 0 or trimap[y, x] == 1:
            continue

        F_p = np.zeros(3, dtype=np.float32)
        B_p = np.zeros(3, dtype=np.float32)

        sum_F = 0.0
        sum_B = 0.0

        alpha_p = refined_alpha[y, x]

        for dy in range(-radius1, radius1 + 1):
            for dx in range(-radius1, radius1 + 1):
                x2 = x + dx
                y2 = y + dy

                if 0 <= x2 < w and 0 <= y2 < h:
                    # NB: Gaussian not normalized, not using confidence
                    Wc_pq = np.exp(-1.0 / sigma_sq1 * (dx * dx + dy * dy))

                    if x != x2 or y != y2:
                        Wc_pq *= abs(refined_alpha[y, x] - refined_alpha[y2, x2])

                    alpha_q = refined_alpha[y2, x2]

                    W_F = Wc_pq * alpha_q
                    W_B = Wc_pq * (1.0 - alpha_q)

                    W_F = max(W_F, 1e-5)
                    W_B = max(W_B, 1e-5)

                    sum_F += W_F
                    sum_B += W_B

                    for c in range(3):
                        F_p[c] += W_F * refined_F[y2, x2, c]
                        B_p[c] += W_B * refined_B[y2, x2, c]

        F_p /= sum_F
        B_p /= sum_B

        final_F[y, x] = F_p
        final_B[y, x] = B_p

        final_alpha[y, x] = estimate_alpha(C_p, F_p, B_p)

        # NB: Not using confidence
        W_FB[y, x] = alpha_p * (1.0 - alpha_p)

    for y, x in pndindex((h, w)):
        C_p = image[y, x]

        if trimap[y, x] == 0 or trimap[y, x] == 1:
            final_confidence[y, x] = trimap[y, x]
            continue

        D_FB = 0.0
        weight_sum = 0.0
        for dy in range(-radius2, radius2 + 1):
            for dx in range(-radius2, radius2 + 1):
                x2 = x + dx
                y2 = y + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    weight_sum += W_FB[y2, x2]
                    D_FB += W_FB[y2, x2] * dist(final_F[y2, x2], final_B[y2, x2])

        D_FB /= weight_sum + 1e-5
        D_FB += 1e-5

        FB_dist = dist(final_F[y, x], final_B[y, x])

        F_p = final_F[y, x]
        B_p = final_B[y, x]

        final_confidence[y, x] = min(1.0, FB_dist / D_FB) * np.exp(-1.0 / sigma_sq2 * np.sqrt(Mp2(C_p, F_p, B_p)))

    for y, x in pndindex((h, w)):
        if trimap[y, x] == 0 or trimap[y, x] == 1:
            final_alpha[y, x] = trimap[y, x]
            continue

        alpha_sum = 0.0
        weight_sum = 0.0
        for dy in range(-radius3, radius3 + 1):
            for dx in range(-radius3, radius3 + 1):
                x2 = x + dx
                y2 = y + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    # NB: Gaussian not normalized, not using final_confidence(x2, y2)
                    D_image_squared = dx * dx + dy * dy
                    is_known = trimap[y2, x2] == 0 or trimap[y2, x2] == 1
                    W_alpha = np.exp(-1.0 / sigma_sq3 * (dx * dx + dy * dy)) + is_known
                    alpha_sum += W_alpha * refined_alpha[y2, x2]
                    weight_sum += W_alpha

        low_frequency_alpha[y, x] = alpha_sum / weight_sum

        final_alpha[y, x] = final_confidence[y, x] * final_alpha[y, x] + (1.0 - final_confidence[y, x]) * low_frequency_alpha[y, x]
