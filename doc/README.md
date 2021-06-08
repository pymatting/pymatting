## Building the docs

1. Install the following packages:
```bash
pip3 install Sphinx sphinxcontrib-bibtex==1.0.0 nbsphinx sphinx_rtd_theme
```
2. Run `./build.sh`
3. The files will appear in `pymatting/build/html`.

## Doc string format

https://numpydoc.readthedocs.io/en/latest/format.html

## Doc string example:

```
    """This function splits the trimap into foreground pixels, background pixels and classifies each pixel as known or unknown. 

    Foreground pixels are pixels where the trimap has value 1.0. Background pixels are pixels where the trimap has value 0.

    Parameters
    ----------
    trimap: array_like
        Trimap with shape :math:`h\\times w`
    flatten: boolean
        If true np.flatten is called on the trimap

    Returns
    -------
    is_fg: np.ndarray
        Boolean array indicating which pixel belongs to the foreground
    is_bg: np.ndarray
        Boolean array indicating which pixel belongs to the background
    is_known: np.ndarray
        Boolean array indicating which pixel is known
    is_unknown: np.ndarray
        Boolean array indicating which pixel is unknown

    """
```
