## Building and viewing the docs

1. Install Python 3.8 or newer (required for [`end_col_offset`](https://docs.python.org/3/whatsnew/3.8.html#ast) and [walrus operator](https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions)).
2. Run the following command in this directory
```bash
python3 build.py
```
3. Navigate to `build/`
4. Run `python3 -m http.server`
5. Visit http://127.0.0.1:8000/

## Doc string format

https://numpydoc.readthedocs.io/en/latest/format.html

## Doc string example:

```python3
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

## Why no Sphinx?

We had to replace Sphinx with our own documentation parser. Sphinx kept injecting new dependencies from third-party servers, which could cause legal issues. In addition, Sphinx kept breaking in mysterious ways. Our new documentation parser can only handle a very limited subset of [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html), but it links all resources statically and is much faster.
