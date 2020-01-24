import os

BUILD_DIR = os.path.join(os.path.expanduser("~"), "tmp")

def get_library_path(name):
    return os.path.join(BUILD_DIR, "pymatting", "solve_" + name + ".so")
