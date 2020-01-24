from pymatting import *
import numpy as np

scale = 1.0

image = load_image("../data/lemur/lemur.png", "RGB", scale, "box")
trimap = load_image("../data/lemur/lemur_trimap.png", "GRAY", scale, "nearest")

# estimate alpha from image and trimap
alpha = estimate_alpha_cf(image, trimap)

# make gray background
new_background = np.zeros(image.shape)
new_background[:, :] = [0.5, 0.5, 0.5]

# estimate foreground from image and alpha
foreground, background = estimate_foreground_ml(image, alpha, return_background=True)

# blend foreground with background and alpha, less color bleeding
new_image = blend(foreground, new_background, alpha)

# save results in a grid
images = [image, trimap, alpha, new_image]
grid = make_grid(images)
save_image("lemur_grid.png", grid)

# save alpha
save_image("lemur_alpha.png", alpha)

# save foreground
save_image("lemur_foreground.png", foreground)

# save background
save_image("lemur_background.png", background)

# save cutout
cutout = stack_images(foreground, alpha)
save_image("lemur_cutout.png", cutout)

# just blending the image with alpha results in color bleeding
color_bleeding = blend(image, new_background, alpha)
grid = make_grid([color_bleeding, new_image])
save_image("lemur_color_bleeding.png", grid)
