from PIL import Image
import numpy as np

def pad(image, pad, fill_value):
    h, w, d = image.shape
    result = np.full((h + 2 * pad, w + 2 * pad, d), fill_value=fill_value, dtype=image.dtype)
    result[pad:pad+h, pad:pad+w] = image
    return result

def main():
    paths = ["image.png", "trimap.png", "alpha.png", "foreground.png"]

    images = [np.array(Image.open(path).convert("RGBA")) for path in paths]
    images = [pad(pad(image, 3, [0, 0, 0, 255]), 50, 255) for image in images]

    a, b, c, d = images

    ab = np.concatenate([a, b], axis=1)
    cd = np.concatenate([c, d], axis=1)
    abcd = np.concatenate([ab, cd], axis=0)

    Image.fromarray(abcd).save("image_grid.png")

if __name__ == "__main__":
    main()
