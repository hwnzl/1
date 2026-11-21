"""Figure 8: anatomical-site classification discrepancies.

Panel (a) gives the absolute disagreement counts, with the diagonal left blank
so that only discordant annotations are shown. Panel (b) divides each cell by
its row total, removing the influence of unequal class sizes across sites.

Usage
-----
python scripts/figure8_confusion_heatmaps.py --csv data/confusion_matrix.csv \
       --out figures/Figure8

Produces .png, .pdf and .tif at 183 mm width, which is the full column width
used by Scientific Data.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "black"


def draw(fig, ax, cax, values, labels, cmap, cbar_label, letter):
    data = np.asarray(values, dtype=float)
    shaded = np.ma.masked_invalid(data)
    colour_map = matplotlib.colormaps[cmap].copy()
    colour_map.set_bad("white")
    image = ax.imshow(shaded, cmap=colour_map, aspect="equal", interpolation="nearest")

    n = data.shape[0]
    high = np.nanmax(data)
    for i in range(n):
        for j in range(n):
            value = data[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=5.4,
                    color="white" if value > high * 0.55 else INK)

    positions = list(range(n))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=5.2)
    ax.set_yticklabels(labels, fontsize=5.2)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=1.6, pad=1.2)
    ax.set_xlabel("Physician B", fontsize=6.5, labelpad=2)
    ax.set_ylabel("Physician A", fontsize=6.5, labelpad=2)
    ax.set_title(letter, loc="left", fontsize=8.5, fontweight="bold", pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    bar = fig.colorbar(image, cax=cax)
    bar.set_label(cbar_label, fontsize=5.8, labelpad=3)
    bar.ax.tick_params(labelsize=5, length=1.6, pad=1.2)
    bar.outline.set_linewidth(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/confusion_matrix.csv")
    parser.add_argument("--out", default="figures/Figure8")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, index_col=0)
    counts = frame.to_numpy(dtype=float)
    counts[np.diag_indices_from(counts)] = np.nan

    proportions = frame.to_numpy(dtype=float)
    proportions = proportions / proportions.sum(axis=1, keepdims=True) * 100
    proportions[np.diag_indices_from(proportions)] = np.nan

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 6,
                         "axes.linewidth": 0.5})

    width, height, side, bottom = 7.2, 3.2, 2.05, 0.86
    gap, bar_width, pad, left = 1.42, 0.10, 0.02, 1.02
    x1 = left
    xc1 = x1 + side + pad
    x2 = xc1 + bar_width + gap
    xc2 = x2 + side + pad

    fig = plt.figure(figsize=(width, height))
    ax1 = fig.add_axes([x1 / width, bottom / height, side / width, side / height])
    cax1 = fig.add_axes([xc1 / width, bottom / height, bar_width / width, side / height])
    ax2 = fig.add_axes([x2 / width, bottom / height, side / width, side / height])
    cax2 = fig.add_axes([xc2 / width, bottom / height, bar_width / width, side / height])

    draw(fig, ax1, cax1, counts, list(frame.index), "Blues",
         "Number of disagreements", "a")
    draw(fig, ax2, cax2, proportions, list(frame.index), "Reds",
         "Disagreement proportion (%)", "b")

    for extension, options in [("png", dict(dpi=600)), ("pdf", {}),
                               ("tif", dict(dpi=600,
                                            pil_kwargs={"compression": "tiff_lzw"}))]:
        fig.savefig(f"{args.out}.{extension}", **options)
    print(f"written {args.out}.png / .pdf / .tif")


if __name__ == "__main__":
    main()
