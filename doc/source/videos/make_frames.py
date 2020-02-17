from pymatting import *
import numpy as np
import os


def write_frames():

    data_dir = "../../../../data/"

    class FrameWriterCallback(object):
        def __init__(self):
            self.n_frames = 0

        def __call__(self, A, x, b, norm_b, r, norm_r):
            alpha = np.clip(x, 0, 1).reshape(trimap.shape)

            save_image("frames/%s/%d.bmp" % (name, self.n_frames), alpha)
            self.n_frames += 1
            print(self.n_frames, norm_r)

    image = load_image(os.path.join(data_dir, "lemur.png"), "RGB")
    trimap = load_image(os.path.join(data_dir, "lemur_trimap.png"), "GRAY")

    for laplacian, name in [
        (cf_laplacian, "cf"),
        (knn_laplacian, "knn"),
        (rw_laplacian, "rw"),
    ]:
        os.makedirs(os.path.join("frames", name), exist_ok=True)

        A, b = make_linear_system(laplacian(image), trimap)

        cg(A, b, callback=FrameWriterCallback(), atol=0.01)

    if 1:
        name = "lkm"
        L_matvec, diag_L = lkm_laplacian(image)

        is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

        lambda_value = 100.0

        c = lambda_value * is_known
        b = lambda_value * is_fg

        def A_matvec(x):
            return L_matvec(x) + c * x

        cg(A_matvec, b, callback=FrameWriterCallback(), atol=0.01)


def make_videos_from_frames():
    for name in os.listdir("frames"):
        directory = os.path.join("frames", name)

        n_frames = len(os.listdir(directory))

        target_length = 10

        fps = n_frames // target_length

        command = f"ffmpeg -y -framerate {fps} -i frames/{name}/%d.bmp {name}.mp4"

        assert 0 == os.system(command)


def remove_frames():
    for name in os.listdir("frames"):
        assert 0 == os.system(f"rm frames/{name}/*.bmp")


def main():
    write_frames()

    make_videos_from_frames()

    remove_frames()


if __name__ == "__main__":
    main()
