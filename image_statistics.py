"""Image-level statistics for the released GastroVL images.

Reports the number of images together with the distribution of width, height
and file size, matching the figures given in "File formats and naming
conventions" of the Data Descriptor.

Usage
-----
python scripts/image_statistics.py --root /path/to/GastroVL
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def iter_images(root):
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="dataset root containing the ten anatomical-site folders")
    parser.add_argument("--ext", default=".png", help="image extension to scan (default: .png)")
    args = parser.parse_args()

    widths, heights, sizes_kb = [], [], []
    for path in iter_images(args.root):
        if path.suffix.lower() != args.ext.lower():
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as exc:                      # unreadable file
            print(f"skipped {path.name}: {exc}")
            continue
        widths.append(w)
        heights.append(h)
        sizes_kb.append(path.stat().st_size / 1024)

    if not widths:
        raise SystemExit("no images found, check --root and --ext")

    w = np.array(widths)
    h = np.array(heights)
    s = np.array(sizes_kb)

    print(f"images            {len(w):>8,}")
    print(f"width, px         {w.min():>8,} - {w.max():,}   median {int(np.median(w)):,}")
    print(f"height, px        {h.min():>8,} - {h.max():,}   median {int(np.median(h)):,}")
    print(f"file size, KB     {s.min():>8,.0f} - {s.max():,.0f}   median {np.median(s):,.0f}")
    print(f"total size, GB    {s.sum() / 1024 / 1024:>8.2f}")


if __name__ == "__main__":
    main()
