If there are new modules in this package you need to run
`sphinx-apidoc -f -o source/ ../pymatting/`
in order to generate the stubs for this module.

After changeing docstrings it is sufficient to call
`make clean && make html`

Doc string format:
https://numpydoc.readthedocs.io/en/latest/format.html

Example doc string:
    """This function splits the trimap into foreground pixels, background pixels and classifies each pixel as known or unknown. 

    Foreground pixels are pixels where the trimap has value 1.0. Background pixels are pixels where the trimap has value 0.

    Parameters
    ----------
    trimap: array_like
        Trimap with shape :math:`h\\times w\\times 1`
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


Fix for less warnings (?):
sphinx-autogen -o generated source/*.rst