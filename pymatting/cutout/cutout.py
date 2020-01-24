from pymatting.util.util import load_image, save_image, stack_images
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml


def cutout(image_path, trimap_path, cutout_path):
    """
    Generate a cutout image from an input image and an input trimap.
    This method is using Closed-Form Alpha Matting as proposed by :cite:`levin2007closed`.

    Parameters
    -----------------
    image_path: str
        Path of input image
    trimap_path: str
        Path of input trimap
    cutout_path: str
        Path of output cutout image
    """
    image = load_image(image_path, "RGB")
    trimap = load_image(trimap_path, "GRAY")

    if image.shape[:2] != trimap.shape[:2]:
        raise ValueError("Input image and trimap must have same size")

    alpha = estimate_alpha_cf(image, trimap)

    foreground = estimate_foreground_ml(image, alpha)

    cutout = stack_images(foreground, alpha)

    save_image(cutout_path, cutout)
