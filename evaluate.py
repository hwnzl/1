"""Evaluate generated descriptions against references, with template-aware strata.

Metrics
-------
BLEU-4 (nltk, method1 smoothing), ROUGE-L (LCS), attribute-level micro-F1 for
location / morphology / colour / size. Size is compared as a coarse category so
that paraphrases ("about 1 cm" vs "1.0 cm") match. Attributes absent from the
references are reported as NaN rather than zero. CIDEr is computed if
pycocoevalcap is available.

Strata
------
All / Normal / Abnormal / Abnormal with reference frequency >=10, 2-9, or 1.
Frequency comes from the full corpus (reference_freq.csv), not the test split.

Usage
-----
    pip install nltk
    python evaluate.py --pred results_zeroshot.csv --freq reference_freq.csv
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

SITES = ["esophagus", "z-line", "cardia", "fundus", "body of stomach",
         "incisura", "antrum", "pylorus", "duodenal bulb", "descending duodenum"]


def tok(text: str) -> list[str]:
    """Lower-case word tokens with punctuation removed.

    Standard practice for BLEU/ROUGE in report-generation benchmarks: without
    this, "mucosa," and "mucosa" count as different tokens and every score
    collapses. Decimal points are preserved ("0.5" stays one token).
    """
    t = (text or "").lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9. ]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = re.sub(r"(?<=[0-9])(?=[a-z])", " ", t)      # "1.5cm" -> "1.5 cm"
    t = re.sub(r"(?<=[a-z])(?=[0-9])", " ", t)
    return [w for w in t.split() if w]


def bleu_n(refs: list, hyps: list, n: int) -> float:
    """Corpus BLEU restricted to n-gram order n (smoothing method1)."""
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
    return corpus_bleu(refs, hyps, weights=weights,
                       smoothing_function=SmoothingFunction().method1)


MORPH = ["ulcer", "erosion", "polyp", "nodul", "elevat", "depress", "flat",
         "erythem", "edema", "oedema", "atroph", "varic", "varix",
         "submucosal", "scar", "diverticul",
         # findings present in the corpus that the first version of the list missed
         "telangiectasia", "plaque", "cheese-like", "debris", "food",
         "foreign body", "defect", "neoplasm", "hyperplas", "swelling",
         "exudate", "purulent", "stricture", "anastomosis", "polypoid"]
COLOR = ["red", "white", "pale", "pink", "yellow"]
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm)")


def size_bucket(text: str) -> set[str]:
    """Map any numeric size to a coarse category so paraphrases match."""
    t = (text or "").lower()
    out = set()
    for val, unit in SIZE_RE.findall(t):
        mm = float(val) * (10 if unit == "cm" else 1)
        if mm < 5:
            out.add("<5mm")
        elif mm < 10:
            out.add("5-10mm")
        elif mm < 20:
            out.add("10-20mm")
        else:
            out.add(">20mm")
    return out


def extract(text: str) -> dict[str, set[str]]:
    t = (text or "").lower()
    return {
        "location": {s for s in SITES if s in t},
        "morphology": {m for m in MORPH if m in t},
        "colour": {c for c in COLOR if c in t},
        "size": size_bucket(t),
    }


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(gen: str, ref: str) -> float:
    g, r = tok(gen), tok(ref)
    if not g or not r:
        return 0.0
    l = lcs_len(g, r)
    p, rc = l / len(g), l / len(r)
    return 2 * p * rc / (p + rc) if p + rc else 0.0


def evaluate_subset(df: pd.DataFrame, cide: bool) -> dict:
    if len(df) == 0:
        return {}
    refs = [[tok(r)] for r in df["reference"]]
    hyps = [tok(g) for g in df["generated"]]
    b1 = bleu_n(refs, hyps, 1)
    b2 = bleu_n(refs, hyps, 2)
    bleu4 = bleu_n(refs, hyps, 4)
    rouge = sum(rouge_l(g, r) for g, r in zip(df["generated"], df["reference"])) / len(df)

    keys = ("location", "morphology", "colour", "size")
    tp = dict.fromkeys(keys, 0)
    fp = dict.fromkeys(keys, 0)
    fn = dict.fromkeys(keys, 0)
    n_ref = dict.fromkeys(keys, 0)      # rows whose reference contains the attribute
    n_gen = dict.fromkeys(keys, 0)      # rows whose generation contains the attribute
    for g, r in zip(df["generated"], df["reference"]):
        eg, er = extract(g), extract(r)
        for k in keys:
            tp[k] += len(eg[k] & er[k])
            fp[k] += len(eg[k] - er[k])
            fn[k] += len(er[k] - eg[k])
            n_ref[k] += 1 if er[k] else 0
            n_gen[k] += 1 if eg[k] else 0

    f1 = {}
    cov = {}
    for k in keys:
        cov[k] = n_ref[k] / len(df)        # share of references that express the attribute
        if n_ref[k] == 0:
            f1[k] = float("nan")           # attribute absent from all references: not evaluable
            continue
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        rc = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        f1[k] = 2 * p * rc / (p + rc) if p + rc else 0.0

    out = {
        "n": len(df),
        "BLEU-1": round(b1, 4),
        "BLEU-2": round(b2, 4),
        "BLEU-4": round(bleu4, 4),
        "ROUGE-L": round(rouge, 4),
        "Loc F1": round(f1["location"], 3),
        "Morph F1": round(f1["morphology"], 3),
        "Colour F1": round(f1["colour"], 3),
        "Size F1": round(f1["size"], 3),
        "ref_morph%": round(cov["morphology"] * 100, 1),
        "ref_colour%": round(cov["colour"] * 100, 1),
        "ref_size%": round(cov["size"] * 100, 1),
        "gen_morph%": round(n_gen["morphology"] / len(df) * 100, 1),
        "gen_size%": round(n_gen["size"] / len(df) * 100, 1),
    }
    if cide:
        out["CIDEr"] = round(cider_score(df), 4)
    return out


def cider_score(df: pd.DataFrame) -> float:
    from pycocoevalcap.cider.cider import Cider
    gts = {i: [r] for i, r in enumerate(df["reference"])}
    res = {i: [g] for i, g in enumerate(df["generated"])}
    score, _ = Cider().compute_score(gts, res)
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results_zeroshot.csv")
    ap.add_argument("--freq", default="reference_freq.csv")
    ap.add_argument("--out", default="metrics.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pred, dtype=str).fillna("")
    df["class"] = df["class"].str.lower()
    print(f"predictions: {len(df)}")

    freq_path = Path(args.freq)
    if freq_path.exists():
        fq = pd.read_csv(freq_path, dtype={"reference": str, "freq": int})
        df = df.merge(fq, on="reference", how="left")
        df["freq"] = df["freq"].fillna(1).astype(int)
        print(f"frequency merged: {len(fq)} references")
    else:
        df["freq"] = 1
        print("WARNING: reference_freq.csv missing, all rows set to freq=1")

    try:
        import pycocoevalcap  # noqa: F401
        cide = True
        print("CIDEr: enabled")
    except ImportError:
        cide = False
        print("CIDEr: skipped (pip install pycocoevalcap)")

    ab = df[df["class"] == "abnormal"]
    subsets = {
        "All": df,
        "Normal": df[df["class"] == "normal"],
        "Abnormal": ab,
        "Abnormal, freq>=10": ab[ab["freq"] >= 10],
        "Abnormal, freq 2-9": ab[(ab["freq"] >= 2) & (ab["freq"] < 10)],
        "Abnormal, freq 1": ab[ab["freq"] == 1],
    }

    rows = []
    for name, sub in subsets.items():
        m = evaluate_subset(sub, cide)
        if m:
            rows.append({"Subset": name, **m})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 180)
    print("\n" + res.to_string(index=False))
    res.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
