import os
from urllib.request import urlopen
import hashlib
import zipfile


def is_pymatting_root():
    return os.path.split(os.getcwd())[-1] == "pymatting" and os.path.isdir("data")


FILES = [
    (
        # url
        "http://www.alphamatting.com/datasets/zip/input_training_lowres.zip",
        # filename
        "data/input_training_lowres.zip",
        # filesize
        18218854,
        # sha256 hash
        "78d403b0671424cd6e6a01b5489a4d26e53c40b1b9b98ea33f3ee02a09ccb5db",
    ),
    (
        "http://www.alphamatting.com/datasets/zip/trimap_training_lowres.zip",
        "data/trimap_training_lowres.zip",
        375510,
        "819a095689057c3747df5d18492b4ef42f13f2d8d58b18de1411938229a3f4b6",
    ),
    (
        "http://www.alphamatting.com/datasets/zip/gt_training_lowres.zip",
        "data/gt_training_lowres.zip",
        2268429,
        "686d560ea2b6e920b97485a34b64c24657b049e751cf97b05c962ebd5f1f7569",
    ),
]


def download_files():
    for url, filename, filesize, sha256_hash in FILES:
        print("Downloading", url)

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                data = f.read()

            if hashlib.sha256(data).hexdigest() == sha256_hash:
                print("File already exists. Download skipped.")
                continue
            else:
                print(
                    "File already exists but SHA256 hash does not match. Downloading anyway."
                )

        with urlopen(url) as r:
            n_bytes = 0
            chunks = []
            while n_bytes < filesize:
                chunk = r.read(10 ** 6)

                if len(chunk) < 0:
                    raise Exception("Failed to download", url)

                n_bytes += len(chunk)
                print("%.2f %%" % (n_bytes * 100.0 / filesize))
                chunks.append(chunk)

        if n_bytes != filesize:
            raise Exception(
                "Failed to download",
                url,
                " Expected",
                filesize,
                "but got",
                n_bytes,
                "bytes.",
            )

        data = b"".join(chunks)

        if hashlib.sha256(data).hexdigest() != sha256_hash:
            raise Exception("SHA256 hash of", url, "does not match")

        with open(filename, "wb") as f:
            f.write(data)


def extract_files():
    for _, filename, _, _ in FILES:
        target_dir = os.path.join(
            "data", os.path.splitext(os.path.split(filename)[-1])[0]
        )

        print("Extracting", filename, "to", target_dir)

        assert os.path.isfile(filename)

        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall(target_dir)


def main():
    if not is_pymatting_root():
        print("Must be run from pymatting root directory")
        return

    download_files()
    extract_files()


if __name__ == "__main__":
    main()
