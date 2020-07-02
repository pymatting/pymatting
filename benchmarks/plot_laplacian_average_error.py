import json
import numpy as np
from plot_results import make_fancy_bar_plot
import matplotlib.pyplot as plt
from config import LABEL_NAMES

with open("results/laplacians.json", "r") as f:
    info = json.load(f)

bar_widths = []
bar_labels = []
for laplacian_name, errors in info.items():
    if laplacian_name not in LABEL_NAMES:
        continue

    bar_labels.append(LABEL_NAMES[laplacian_name])

    errors = list(errors.values())

    bar_widths.append(np.mean(errors))

bar_widths, bar_labels = zip(*sorted(zip(bar_widths, bar_labels), reverse=True))

xticks = np.linspace(0, 70, 8)

plt.figure(figsize=(6, 2))
make_fancy_bar_plot("results/laplacians", None, None, xticks, bar_widths, bar_labels)
