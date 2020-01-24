import os
import argparse


def is_pymatting_root():
    return os.path.split(os.getcwd())[-1] == "pymatting" and os.path.isdir("data")


TESTS = [
    "test_util.py",
    "test_simple_api.py",
    "test_ichol.py",
    "test_lkm.py",
    "test_estimate_alpha.py",
    "test_kdtree.py",
    "test_preconditioners.py",
    "test_laplacians.py",
    "test_boxfilter.py",
    "test_foreground.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_gpu", help="exclude tests for GPU implementation", action="store_true"
    )
    args = parser.parse_args()
    sucess_counter = 0

    if not is_pymatting_root():
        print("Must be run from pymatting root directory")
        return

    for filename in TESTS:
        filename = os.path.join("tests", filename)
        print("Running", filename)
        command = "/usr/bin/env python3 " + filename

        if filename == "tests/test_foreground.py" and args.no_gpu:
            command += " --no_gpu"

        err = os.system(command)
        if err != 0:
            print("Test", filename, "failed")
        else:
            sucess_counter += 1
            print("Test", filename, "succeeded")

    print(sucess_counter, "tests out of", len(TESTS), "passed")


if __name__ == "__main__":
    main()
