import re
import os
import pathlib
import hashlib
from urllib.request import urlretrieve

mapping = {
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js": "_static/require.min.js",
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js": "_static/tex-mml-chtml.js",
    "https://github.blog/wp-content/uploads/2008/12/forkme_right_gray_6d6d6d.png?resize=149%2C149": "_static/forkme_right_gray_6d6d6d.png",
    "https://github.com/pymatting/videos/blob/master/cf_web.mp4?raw=true": "_static/cf_web.mp4",
    "https://github.com/pymatting/videos/blob/master/knn_web.mp4?raw=true": "_static/knn_web.mp4",
    "https://github.com/pymatting/videos/blob/master/lkm_web.mp4?raw=true": "_static/lkm_web.mp4",
    "https://github.com/pymatting/videos/blob/master/rw_web.mp4?raw=true": "_static/rw_web.mp4",
}

fonts = """
MathJax_Zero.woff
MathJax_Main-Regular.woff
MathJax_Main-Bold.woff
MathJax_Math-Italic.woff
MathJax_Main-Italic.woff
MathJax_Math-BoldItalic.woff
MathJax_Size1-Regular.woff
MathJax_Size2-Regular.woff
MathJax_Size3-Regular.woff
MathJax_Size4-Regular.woff
MathJax_AMS-Regular.woff
MathJax_Calligraphic-Regular.woff
MathJax_Calligraphic-Bold.woff
MathJax_Fraktur-Regular.woff
MathJax_Fraktur-Bold.woff
MathJax_SansSerif-Regular.woff
MathJax_SansSerif-Bold.woff
MathJax_SansSerif-Italic.woff
MathJax_Script-Regular.woff
MathJax_Typewriter-Regular.woff
MathJax_Vector-Regular.woff
MathJax_Vector-Bold.woff
"""

def main():
    # Prepare font download directory
    font_directory = "_static/output/chtml/fonts/woff-v2/"
    os.makedirs("build/html/" + font_directory, exist_ok=True)

    # Add fonts to mappings
    for font in fonts.strip().split("\n"):
        url = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2/" + font
        filename = font_directory + font
        mapping[url] = filename

    # Download files to static file directory
    for url, filename in mapping.items():
        filename = "build/html/" + filename

        if os.path.isfile(filename):
            print(filename, "already exists. Download skipped")
        else:
            print("Downloading", url, "to", filename)
            urlretrieve(url, filename)

    def replace_url(match):
        url = match.group(1)

        # Keep _static and _image urls
        if url.startswith("_static"):
            new_url = url
        elif url.startswith("_images"):
            new_url = url
        elif url in mapping:
            # Map URLs specified in mapping dict
            new_url = mapping[url]
        else:
            # Complain if someone introduced new external dependencies.
            raise ValueError("Detected new external url " + url + ". Please add mapping in " + __file__)

        return 'src="' + new_url + '"'

    # For each html file
    for path in pathlib.Path("build/html").rglob("*.html"):

        text = path.read_text()

        # Replace URLs
        text = re.sub('src="(.*?)"', replace_url, text)
        # Remove attributes only required for external URLs
        text = re.sub('crossorigin="anonymous"', "", text)
        text = re.sub('integrity=".*?"', "", text)

        path.write_text(text)

if __name__ == "__main__":
    main()
