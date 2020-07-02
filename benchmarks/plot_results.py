import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

names = {
    "cg_icholt": "Ours",
    "pyamg": "PyAMG",
    "petsc": "PETSc",
    "amgcl": "AMGCL",
    "eigen_cholesky": "Eigen $L D L^T$",
    "mumps": "MUMPS",
    "umfpack": "UMFPACK",
    "superlu": "SuperLU",
}


def make_fancy_bar_plot(
    filename, title, xlabel, xticks, bar_widths, bar_labels, xerr=None
):
    plt.barh(bar_labels, bar_widths, alpha=0.5)

    if xerr is None:
        xerr = [0] * len(bar_widths)

    for y, (x, err) in enumerate(zip(bar_widths, xerr)):
        if 0:
            x0 = x - err
            x1 = x + err

            r = 0.1
            plt.plot([x0, x1], [y, y], color="black", alpha=0.5)
            plt.plot([x0, x0], [y - r, y + r], color="black", alpha=0.5)
            plt.plot([x1, x1], [y - r, y + r], color="black", alpha=0.5)
        else:
            x1 = x

        s = "{:g}".format(float("{:.{p}g}".format(x, p=3)))

        plt.text(x1 + 0.01 * np.max(bar_widths), y, s, va="center")

    plt.xticks(xticks)
    # plt.xlabel(xlabel)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(filename + ext)
    plt.show()


def plot_memory_usage():
    max_scale = max(result["scale"] for result in solver_results[0])

    labels = []
    peak_memory_usage = []
    for results in solver_results:
        solver_name = results[0]["solver_name"]

        if solver_name not in names:
            continue

        labels.append(solver_name)

        mbs = []
        for result in results:
            if result["scale"] != max_scale:
                continue

            mb = 1e-6 * np.float64(result["memory_usage"])

            mbs.append(mb.max())

        peak_memory_usage.append(mbs)

    labels = np.array([names[label] for label in labels])
    avg_peak_memory_usage = np.array([np.mean(x) for x in peak_memory_usage])
    std_peak_memory_usage = np.array([np.std(x) for x in peak_memory_usage])

    indices = np.argsort(avg_peak_memory_usage)[::-1]

    labels = labels[indices]
    avg_peak_memory_usage = avg_peak_memory_usage[indices]

    make_fancy_bar_plot(
        "results/average_peak_memory_usage",
        "Solver Peak Memory Usage",
        "Memory Usage [MB]",
        np.arange(0, 9001, 1000),
        avg_peak_memory_usage,
        labels,
        std_peak_memory_usage,
    )


def plot_time():
    max_scale = max(result["scale"] for result in solver_results[0])

    labels = []
    rows = []
    for results in solver_results:
        solver_name = results[0]["solver_name"]

        if solver_name not in names:
            continue

        labels.append(solver_name)

        row = []
        for result in results:
            if result["scale"] != max_scale:
                continue

            row.append(result["build_time"] + result["solve_time"])

        rows.append(row)

    labels = np.array([names[label] for label in labels])
    avg_row = np.array([np.mean(row) for row in rows])
    std_row = np.array([np.std(row) for row in rows])

    indices = np.argsort(avg_row)[::-1]

    labels = labels[indices]
    avg_row = avg_row[indices]
    std_row = std_row[indices]

    make_fancy_bar_plot(
        "results/average_running_time",
        "Solver Average Running Time",
        "Time [seconds]",
        np.arange(0, 301, 50),
        avg_row,
        labels,
        std_row,
    )


def plot_time_image_size():
    plt.ylabel("Time [seconds]")
    plt.xlabel("Image size [pixels]")

    sort_values = []

    for results in solver_results:
        solver_name = results[0]["solver_name"]

        if solver_name not in names:
            continue

        timings = defaultdict(list)
        n_pixels = defaultdict(list)
        for result in results:
            width = result["width"]
            height = result["height"]

            n = width * height

            timings[result["scale"]].append(result["build_time"] + result["solve_time"])
            n_pixels[result["scale"]].append(n)

        scales = sorted(n_pixels.keys())
        n_pixels = [np.mean(n_pixels[scale]) for scale in scales]
        timings = [np.mean(timings[scale]) for scale in scales]

        sort_values.append(timings[-1])

        plt.semilogy(n_pixels, timings, label=names[solver_name])

    indices = np.argsort(sort_values)

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend([handles[i] for i in indices], [labels[i] for i in indices])

    plt.tight_layout()
    filename = "results/time_image_size"
    for ext in [".png", ".pdf"]:
        plt.savefig(filename + ext)
    plt.show()


if __name__ == "__main__":
    directory = "results/solver"

    solver_results = []
    for solver in os.listdir(directory):
        path = os.path.join(directory, solver)
        with open(path, "rb") as f:
            solver_results.append(json.load(f))

    plot_time_image_size()
    plot_memory_usage()
    plot_time()
