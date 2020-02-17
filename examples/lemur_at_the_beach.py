from pymatting import *

scale = 1.0

image = load_image("../data/lemur/lemur.png", "RGB", scale, "box")
trimap = load_image("../data/lemur/lemur_trimap.png", "GRAY", scale, "nearest")

# estimate alpha from image and trimap
alpha = estimate_alpha_cf(image, trimap)

# load new background
new_background = load_image("../data/lemur/beach.png", "RGB", scale, "box")

# estimate foreground from image and alpha
foreground, background = estimate_foreground_ml(image, alpha, return_background=True)

# blend foreground with background and alpha
new_image = blend(foreground, new_background, alpha)

# save results in a grid
images = [image, trimap, alpha, new_image]
grid = make_grid(images)
save_image("lemur_at_the_beach.png", grid)
