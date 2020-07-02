import matplotlib.pyplot as plt
import json
import numpy as np
from natsort import natsorted
from config import LABEL_NAMES

with open("results/laplacians.json", "r") as f:
    items = json.load(f)

COLORS = [
    "#" + color
    for color in """
    ff8080
    ffae80
    ffdd80
    f5ff80
    80ffb7
    80fffd
    80d0ff
    8097ff
    c680ff
    ff80ca
    ff8080
""".split()
]

edgecolor = "black"

colors = np.delete(COLORS, [0, 2])

bar_scale = 0.8

indices = 1 + np.arange(27)

plt.figure(figsize=(11, 4))

print(items)

items = dict((key, value) for key, value in items.items() if key in LABEL_NAMES)

laplacian_names = natsorted(items.keys())

for i, laplacian_name in enumerate(laplacian_names):
    err = list(items[laplacian_name][str(j)] for j in indices)

    offset = (i - (len(laplacian_names) - 1) / 2) * bar_scale / len(laplacian_names)
    x = np.arange(1, 1 + len(indices)) + offset

    for color, alpha, use_label in [
        ("none", 1.0, False),
        (None, 0.5, True),
    ]:
        plt.bar(
            x=x,
            height=err,
            width=bar_scale / len(laplacian_names),
            label=LABEL_NAMES[laplacian_name] if use_label else None,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=0.5,
        )

    plt.xticks(indices, indices)

pad = 0.1
plt.xlim([0.5 - pad, 26.5 + pad])
plt.legend()
plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig("results/laplacian_quality_many_bars" + ext)
plt.show()
