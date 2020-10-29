import os
from setuptools import setup, find_packages


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


directory = os.path.dirname(os.path.abspath(__file__))

# load information about package
path = os.path.join(directory, "pymatting", "__about__.py")
about = {}
exec(load_text(path), about)

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
    # Fix for Numba caching issue
    zip_safe=False,
)
