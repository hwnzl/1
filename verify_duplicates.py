"""Full duplicate audit: MD5 -> pHash -> SSIM, single pass over the images.

Writes v2_* outputs so the previous audit results are preserved for comparison.
"""
import csv
import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import uniform_filter

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(os.environ.get("GASTROVL_ROOT", "GastroVL"))
OUT = Path(os.environ.get("GASTROVL_OUT", "outputs"))
HAMMING_MAX = 5
SSIM_SIZE = 256


def phash(path: Path, hash_size: int = 8, factor: int = 4) -> int:
    size = hash_size * factor
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size]
    med = np.median(low[1:, :])
    value = 0
    for b in (low > med).flatten():
        value = (value << 1) | int(b)
    return value


def ssim_of(a_path: Path, b_path: Path) -> float:
    a = np.asarray(Image.open(a_path).convert("L").resize((SSIM_SIZE, SSIM_SIZE),
                                                          Image.BILINEAR), dtype=np.float64)
    b = np.asarray(Image.open(b_path).convert("L").resize((SSIM_SIZE, SSIM_SIZE),
                                                          Image.BILINEAR), dtype=np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = uniform_filter(a, 11), uniform_filter(b, 11)
    saa = uniform_filter(a * a, 11) - mu_a ** 2
    sbb = uniform_filter(b * b, 11) - mu_b ** 2
    sab = uniform_filter(a * b, 11) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    return float((num / den).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in ROOT.rglob("*.png") if f.is_file())
    n = len(files)
    print(f"images: {n}", flush=True)

    # ---------- stage 1: MD5 exact duplicates ----------
    t0 = time.time()
    by_size = defaultdict(list)
    for f in files:
        by_size[f.stat().st_size].append(f)
    exact_groups = []
    for size, group in by_size.items():
        if len(group) < 2:
            continue
        digest = defaultdict(list)
        for f in group:
            digest[hashlib.md5(f.read_bytes()).hexdigest()].append(f)
        for h, g in digest.items():
            if len(g) > 1:
                exact_groups.append((h, size, g))
    exact_imgs = sum(len(g) for _, _, g in exact_groups)
    print(f"[MD5] exact groups: {len(exact_groups)}, images involved: {exact_imgs} "
          f"({exact_imgs / n * 100:.2f}%)  {time.time() - t0:.0f}s", flush=True)

    with open(OUT / "v2_exact_duplicates.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["group_id", "md5", "bytes", "n_files", "file_paths"])
        for i, (h, sz, g) in enumerate(sorted(exact_groups, key=lambda x: -len(x[2])), 1):
            w.writerow([f"E{i}", h, sz, len(g), ";".join(str(p.relative_to(ROOT)) for p in g)])

    # ---------- stage 2: pHash ----------
    t0 = time.time()
    hashes = []
    failed = []
    for i, f in enumerate(files, 1):
        try:
            hashes.append(phash(f))
        except Exception as exc:                      # noqa: BLE001
            failed.append((f, str(exc)))
            hashes.append(None)
        if i % 2000 == 0:
            print(f"  phash {i}/{n}  {time.time() - t0:.0f}s", flush=True)

    valid_idx = [i for i, h in enumerate(hashes) if h is not None]
    h_arr = np.array([hashes[i] for i in valid_idx], dtype=np.uint64)
    packed = h_arr.view(np.uint8).reshape(len(valid_idx), 8)
    pop = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

    candidates = []
    block = 512
    for start in range(0, len(valid_idx), block):
        a = packed[start:start + block]
        dist = pop[a[:, None, :] ^ packed[None, :, :]].sum(axis=2)
        for i, j in zip(*np.where(np.triu(dist, k=1) > 0)):
            d = int(dist[i, j])
            if d <= 8:                                # keep a slightly wider net for Stage 3
                gi, gj = valid_idx[start + i], valid_idx[j]
                if gj > gi:
                    candidates.append((gi, gj, d))
    print(f"[pHash] pairs <=8: {len(candidates)}   pairs <=5: "
          f"{sum(1 for *_, d in candidates if d <= HAMMING_MAX)}  {time.time() - t0:.0f}s", flush=True)

    # ---------- stage 3: SSIM on Hamming <= 5 ----------
    t0 = time.time()
    cache: dict[int, np.ndarray] = {}

    def gray(i: int) -> np.ndarray:
        if i not in cache:
            cache[i] = np.asarray(
                Image.open(files[i]).convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR),
                dtype=np.float64)
        return cache[i]

    def ssim_cached(i: int, j: int) -> float:
        a, b = gray(i), gray(j)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu_a, mu_b = uniform_filter(a, 11), uniform_filter(b, 11)
        saa = uniform_filter(a * a, 11) - mu_a ** 2
        sbb = uniform_filter(b * b, 11) - mu_b ** 2
        sab = uniform_filter(a * b, 11) - mu_a * mu_b
        num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
        den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
        return float((num / den).mean())

    checked = []
    todo = [(i, j, d) for i, j, d in candidates if d <= HAMMING_MAX]
    for k, (i, j, d) in enumerate(todo, 1):
        checked.append((i, j, d, ssim_cached(i, j)))
        if k % 300 == 0:
            print(f"  ssim {k}/{len(todo)}", flush=True)

    with open(OUT / "v2_near_duplicates_ssim.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["path_a", "path_b", "hamming", "ssim"])
        for i, j, d, s in sorted(checked, key=lambda x: -x[3]):
            w.writerow([str(files[i].relative_to(ROOT)), str(files[j].relative_to(ROOT)), d, round(s, 4)])

    # ---------- summary ----------
    lines = [f"images scanned: {n}",
             f"unreadable files: {len(failed)}"]
    for f, err in failed:
        lines.append(f"  {f.relative_to(ROOT)}  -> {err}")
    lines.append("")
    lines.append(f"[MD5]  exact duplicate groups: {len(exact_groups)}  "
                 f"images involved: {exact_imgs} ({exact_imgs / n * 100:.2f}%)")
    lines.append(f"[pHash] candidate pairs (Hamming <=5): {len(todo)}")
    for thr in (0.90, 0.95, 0.98):
        sel = [x for x in checked if x[3] >= thr]
        inv = {i for i, *_ in sel} | {j for _, j, *_ in sel}
        lines.append(f"[SSIM] >= {thr}: {len(sel)} pairs, {len(inv)} images "
                     f"({len(inv) / n * 100:.2f}%)")
    if checked:
        lines.append(f"[SSIM] max = {max(x[3] for x in checked):.4f}")
    text = "\n".join(lines)
    (OUT / "v2_audit_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
