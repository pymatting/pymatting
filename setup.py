import os
import pkgutil
from setuptools import setup, find_packages
from numba.pycc import CC


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


directory = os.path.dirname(os.path.abspath(__file__))

# load information about package
path = os.path.join(directory, "pymatting", "__about__.py")
about = {}
exec(load_text(path), about)

# collect AOT-compiled modules
cc = CC("aot")
for importer, module_name, _ in pkgutil.walk_packages(
    [os.path.join(directory, "pymatting_aot")]
):
    if module_name != "cc":
        module = importer.find_module(module_name).load_module(module_name)
        for function_name, (function, signature) in module.exports.items():
            cc.export(function_name, signature)(function)

setup(
    name=about["__title__"],
    version=about["__version__"],
    url=about["__uri__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description=about["__summary__"],
    long_description=load_text("README.md"),
    long_description_content_type="text/markdown",
    license=about["__license__"],
    packages=find_packages(),
    install_requires=load_text("requirements.txt").strip().split("\n"),
    keywords="alpha matting",
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={"Source": "https://github.com/pymatting/pymatting"},
    ext_modules=[cc.distutils_extension()],
    # Fix for Numba caching issue
    zip_safe=False,
)
