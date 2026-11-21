"""Figure 9: site-specific consistency in the normal/abnormal classification.

For each anatomical site the bar gives the number of images on which the two
annotators agreed and disagreed, with the percentage of the site total in
brackets. Sites are ordered by the number of discordant images.

Usage
-----
python scripts/figure9_site_consistency.py --csv data/site_agreement.csv \
       --out figures/Figure9
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/site_agreement.csv")
    parser.add_argument("--out", default="figures/Figure9")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    frame = frame.sort_values("discordant", ascending=True).reset_index(drop=True)

    images = frame["images"].to_numpy(dtype=float)
    discordant = frame["discordant"].to_numpy(dtype=float)
    consistent = images - discordant
    consistent_pct = consistent / images * 100
    discordant_pct = discordant / images * 100

    positions = np.arange(len(frame))

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    ax.barh(positions, consistent, color="#2c7fb8", edgecolor="black",
            linewidth=0.5, label="Consistent")
    ax.barh(positions, discordant, left=consistent, color="#fd8d3c",
            edgecolor="black", linewidth=0.5, label="Disagreement")

    for i in range(len(frame)):
        ax.text(consistent[i] / 2, positions[i],
                f"{consistent[i]:,.0f}\n({consistent_pct[i]:.1f}%)",
                ha="center", va="center", color="white", fontsize=7)
        label = f"{discordant[i]:,.0f} ({discordant_pct[i]:.1f}%)"
        ax.text(images[i] * 1.01, positions[i], label, ha="left", va="center",
                color="black", fontsize=7)

    ax.set_yticks(positions)
    ax.set_yticklabels(frame["site"], fontsize=8)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_xlabel("Number of images", fontsize=9)
    ax.set_ylabel("")
    ax.set_xlim(0, images.max() * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
              frameon=False, fontsize=8)

    fig.tight_layout()
    for extension, options in [("png", dict(dpi=600)), ("pdf", {}),
                               ("tif", dict(dpi=600,
                                            pil_kwargs={"compression": "tiff_lzw"}))]:
        fig.savefig(f"{args.out}.{extension}", **options)
    print(f"written {args.out}.png / .pdf / .tif")


if __name__ == "__main__":
    main()
