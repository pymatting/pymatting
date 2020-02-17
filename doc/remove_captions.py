import sys
import re

name = sys.argv[1]

with open(name, "r") as f:
    text = f.read()

text = re.sub(r"Module contents\n-+\n", "", text)
text = re.sub(r"Submodules\n-+\n", "", text)
text = re.sub(r"Subpackages\n-+\n", "", text)

with open(name, "w") as f:
    f.write(text)
