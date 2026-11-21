"""Prepare a compact upload bundle for GPU-based baseline evaluation.

Selects the test split, resizes images to a fixed long side, copies the
matching reference texts, and archives everything into one tar.gz that can
be dragged into JupyterLab in a few minutes instead of uploading 22 GB.

Usage
-----
    python prepare_gpu_bundle.py                 # test split, long side 512
    python prepare_gpu_bundle.py --long-side 768
    python prepare_gpu_bundle.py --split test val
"""
from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DATA_ROOT = Path(os.environ.get("GASTROVL_ROOT", "GastroVL"))
INDEX = DATA_ROOT / "index.xlsx"
SPLIT = Path(os.environ.get("GASTROVL_SPLIT", "outputs/official_split.csv"))
BASE = Path(os.environ.get("GASTROVL_BASE", "."))
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-side", type=int, default=512)
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--format", default="PNG", choices=["PNG", "JPEG"])
    ap.add_argument("--name", default="gpu_bundle", help="输出目录与压缩包名称")
    args = ap.parse_args()

    WORK = BASE / args.name
    ARCHIVE = BASE / f"{args.name}.tar.gz"

    idx = pd.read_excel(INDEX, dtype=str)
    idx.columns = [c.strip() for c in idx.columns]
    sp = pd.read_csv(SPLIT, dtype=str)
    df = idx.merge(sp, on=["image_id", "patient_id"], how="left")
    print("index rows:", len(df), " | missing split:", df["split"].isna().sum())

    sel = df[df["split"].isin(args.split)].reset_index(drop=True)
    print(f"selected {len(sel)} images for split(s): {args.split}")
    meta_name = args.split[0] + ".csv" if len(args.split) == 1 else "meta.csv"

    img_dir = WORK / "images"
    txt_dir = WORK / "texts"
    for d in (img_dir, txt_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    failed = []
    for i, row in sel.iterrows():
        src = DATA_ROOT / row["image_path"]
        dst = img_dir / f"{row['image_id']}.{args.format.lower()}"
        try:
            im = Image.open(src).convert("RGB")
            w, h = im.size
            s = args.long_side / max(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            im.save(dst, optimize=True, quality=92) if args.format == "JPEG" else im.save(dst, optimize=True)
        except Exception as exc:                       # noqa: BLE001
            failed.append((row["image_id"], str(exc)))
            continue
        tsrc = DATA_ROOT / row["text_path"]
        if tsrc.exists():
            shutil.copy(tsrc, txt_dir / f"{row['image_id']}.txt")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sel)}", flush=True)

    meta = sel[["image_id", "patient_id", "anatomical_site", "class", "split"]].copy()
    meta.to_csv(WORK / meta_name, index=False, encoding="utf-8-sig")

    total = sum(f.stat().st_size for f in img_dir.iterdir())
    print(f"\nimages: {len(list(img_dir.iterdir()))}   total size: {total/1e6:.1f} MB")
    if failed:
        print(f"failed: {len(failed)} -> {failed[:5]}")

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(WORK / "images", arcname=f"{args.name}/images")
        tar.add(WORK / "texts", arcname=f"{args.name}/texts")
        tar.add(WORK / meta_name, arcname=f"{args.name}/{meta_name}")
    size = ARCHIVE.stat().st_size / 1e6
    print(f"\narchive: {ARCHIVE}  ({size:.1f} MB)")
    print("upload this single file to the cloud instance, then run:")
    print("  tar -xzf gpu_bundle.tar.gz && ls gpu_bundle")


if __name__ == "__main__":
    main()
