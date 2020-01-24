import json
import numpy as np
from plot_results import make_fancy_bar_plot
import matplotlib.pyplot as plt

names = {
    "rw_laplacian": "Random Walk",
    "lbdm_laplacian": "Learning-Based",
    "knn_laplacian": "KNN",
    "cf_laplacian": "Closed-Form",
}

with open("results/laplacians.json", "r") as f:
    info = json.load(f)

bar_widths = []
bar_labels = []
for laplacian_name, errors in info.items():
    bar_labels.append(names[laplacian_name])
    
    errors = list(errors.values())
    
    bar_widths.append(np.mean(errors))

bar_widths, bar_labels = zip(*sorted(zip(bar_widths, bar_labels), reverse=True))

xticks = np.linspace(0, 11, 12)

plt.figure(figsize=(6, 2))
make_fancy_bar_plot("results/laplacians", None, None, xticks, bar_widths, bar_labels)
