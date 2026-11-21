"""Analyse the text-description audit exported by audit_rating.html.

Statistics, in plain terms
--------------------------
Every answered cell is a categorical variable. Each summary metric is a
proportion, and every proportion needs three things stated explicitly:
numerator, denominator, and a confidence interval.

    metric                     numerator                    denominator
    ---------------------------------------------------------------------
    factual consistency        rows rated "full"            all rows
    unsupported statements     rows rated "unsupported"     all rows
    feature accuracy (strict)  rows rated "ok"              rows rated ok/part/bad
    feature accuracy (lenient) ok + 0.5 x part              rows rated ok/part/bad
    omission rate              rows not rated "no"          all rows

"Not applicable" never enters a denominator: size and margin are unassessable
for diffuse mucosal change and for non-lesion abnormalities, and counting those
as errors would understate accuracy.

Confidence intervals are obtained by bootstrapping **patients**, not rows,
because up to two pairs per patient are audited and rows from one patient are
not independent.

Usage
-----
    python scripts/agreement_statistics.py --r1 audit_R1.csv --r2 audit_R2.csv
    python scripts/agreement_statistics.py --r1 audit_R1.csv --boot 2000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("GASTROVL_BASE", "."))
DEFAULT_SAMPLE = BASE / "outputs" / "audit" / "audit_sample.csv"

FEATS = [("location", "Location"), ("morphology", "Morphology"),
         ("size", "Size"), ("margin", "Margin"), ("colour", "Colour")]
FEAT_VALUES = ("ok", "part", "bad")


def ac1(a: list[int], b: list[int], k: int) -> float:
    """Gwet's AC1: agreement coefficient that stays stable when one category dominates."""
    n = len(a)
    if n == 0:
        return float("nan")
    ca = np.bincount(a, minlength=k) / n
    cb = np.bincount(b, minlength=k) / n
    pi = (ca + cb) / 2
    used = int((pi > 0).sum())
    if used < 2:
        return float("nan")
    po = float((np.array(a) == np.array(b)).mean())
    pe = float((pi * (1 - pi)).sum() / (used * (used - 1)))
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def pabak(a: list[int], b: list[int]) -> float:
    """Prevalence-adjusted bias-adjusted kappa: 2*Po - 1."""
    if not len(a):
        return float("nan")
    po = float((np.array(a) == np.array(b)).mean())
    return 2 * po - 1


def fmt(x: float, digits: int = 3) -> str:
    return "n/a" if (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def kappa(a: list[int], b: list[int], k: int, weighted: bool = True) -> float:
    """Cohen's kappa. Linear weights when the scale is ordinal, unweighted when nominal."""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x, y] += 1
    ra = obs.sum(1)
    cb = obs.sum(0)
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            if weighted:
                w = 1 - abs(i - j) / (k - 1)
            else:
                w = 1.0 if i == j else 0.0
            num += w * obs[i, j]
            den += w * ra[i] * cb[j] / n
    return (num - den) / (n - den) if n != den else float("nan")


def boot_ci(df: pd.DataFrame, stat, B: int, seed: int = 1) -> tuple[float, float, float]:
    """Point estimate plus 95% CI from a patient-clustered bootstrap."""
    point = stat(df)
    pids = df["patient_id"].dropna().unique()
    if len(pids) < 2 or B <= 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("patient_id")}
    vals = []
    for _ in range(B):
        pick = rng.choice(pids, size=len(pids), replace=True)
        sample = pd.concat([groups[p] for p in pick], ignore_index=True)
        v = stat(sample)
        if not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def pct(mask: pd.Series, denom_mask: pd.Series | None = None) -> float:
    """Percentage of True in `mask`, optionally restricted to `denom_mask`."""
    if denom_mask is None:
        return float(mask.mean() * 100) if len(mask) else float("nan")
    if not int(denom_mask.sum()):
        return float("nan")
    return float(mask[denom_mask].mean() * 100)


def load_ratings(path: str, sheet: str = "") -> pd.DataFrame:
    """Read a CSV exported by the rating page, or one sheet of the Excel template."""
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
        except ValueError:
            df = pd.read_excel(path, sheet_name=0, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df = df.fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--r2", default="")
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (0 to disable)")
    ap.add_argument("--out", default=str(BASE / "outputs" / "audit" / "audit_metrics.csv"))
    args = ap.parse_args()

    sample = pd.read_csv(args.sample, dtype=str).fillna("")
    r1 = load_ratings(args.r1, "R1")
    if "DEMO" in str(r1.get("rater", pd.Series([""])).iloc[0]).upper() or "SYNTHETIC" in args.r1.upper():
        print("!" * 70)
        print("!! SIMULATED DATA — for pipeline testing only, never for reporting")
        print("!" * 70)
        print()
    r1 = r1.merge(sample[["pair_id", "patient_id", "site", "class"]], on="pair_id", how="left")

    print(f"rater 1: {len(r1)} pairs from {r1.patient_id.nunique()} patients")
    print(f"class split: {r1['class'].value_counts().to_dict()}")
    print()
    rows = []

    # ---------- 1. factual consistency ----------
    print("=== 1. Factual consistency (by class) ===")
    print(f"{'class':10s} {'n':>4s} {'fully%':>16s} {'partially%':>8s} {'unsupported%':>14s}")
    for cls in ["normal", "abnormal"]:
        sub = r1[r1["class"] == cls]
        if not len(sub):
            continue
        p, lo, hi = boot_ci(sub, lambda d: pct(d["consistency"] == "full"), args.boot)
        print(f"{cls:10s} {len(sub):4d} {p:7.1f} ({lo:.1f}-{hi:.1f})"
              f" {pct(sub['consistency'] == 'partial'):11.1f}"
              f" {pct(sub['consistency'] == 'unsupported'):13.1f}")
        rows += [
            {"metric": "consistency_full_pct", "subset": cls, "value": round(p, 1),
             "ci_low": round(lo, 1), "ci_high": round(hi, 1), "n": len(sub)},
            {"metric": "consistency_partial_pct", "subset": cls,
             "value": round(pct(sub["consistency"] == "partial"), 1), "n": len(sub)},
            {"metric": "consistency_unsupported_pct", "subset": cls,
             "value": round(pct(sub["consistency"] == "unsupported"), 1), "n": len(sub)},
        ]
    print()

    # ---------- 2. feature accuracy (abnormal only) ----------
    print("=== 2. Feature accuracy (abnormal pairs; N/A excluded from the denominator) ===")
    print(f"{'attribute':12s} {'n':>4s} {'strict%':>16s} {'lenient%':>10s}")
    ab = r1[r1["class"] == "abnormal"]
    for key, label in FEATS:
        usable = ab[ab[key].isin(FEAT_VALUES)]
        if not len(usable):
            print(f"{label:12s} {0:4d} {'n/a':>16s} {'n/a':>10s}")
            continue
        strict = lambda d, k=key: pct(d[k] == "ok", d[k].isin(FEAT_VALUES))
        lenient = lambda d, k=key: float((d[k].isin(["ok", "part"]) *
                                          np.where(d[k] == "ok", 1.0, 0.5))[d[k].isin(FEAT_VALUES)].mean() * 100)
        p, lo, hi = boot_ci(usable, strict, args.boot)
        lp = lenient(usable)
        print(f"{label:12s} {len(usable):4d} {p:7.1f} ({lo:.1f}-{hi:.1f}) {lp:9.1f}")
        rows += [
            {"metric": f"feature_{key}_strict_pct", "subset": "abnormal", "value": round(p, 1),
             "ci_low": round(lo, 1), "ci_high": round(hi, 1), "n": len(usable)},
            {"metric": f"feature_{key}_lenient_pct", "subset": "abnormal",
             "value": round(lp, 1), "n": len(usable)},
        ]
    print()
    if len(ab):
        for k in ["focal", "diffuse", "other", "none"]:
            v = pct(ab["lesion"] == k)
            print(f"  lesion type {k:8s} {v:5.1f}%")
            rows.append({"metric": f"lesion_type_{k}_pct", "subset": "abnormal",
                         "value": round(v, 1), "n": len(ab)})
    print()

    # ---------- 3. omission ----------
    print("=== 3. Omission of salient findings ===")
    for cls in ["normal", "abnormal"]:
        sub = r1[r1["class"] == cls]
        if not len(sub):
            continue
        p, lo, hi = boot_ci(sub, lambda d: pct(d["omission"] != "no"), args.boot)
        print(f"  {cls:9s} n={len(sub):3d}  omission {p:5.1f}% ({lo:.1f}-{hi:.1f})"
              f"   {sub['omission'].value_counts().to_dict()}")
        rows.append({"metric": "omission_pct", "subset": cls, "value": round(p, 1),
                     "ci_low": round(lo, 1), "ci_high": round(hi, 1), "n": len(sub)})
    print()

    # ---------- 4. inter-rater agreement ----------
    if args.r2:
        r2 = load_ratings(args.r2, "R2")
        m = r1.merge(r2, on="pair_id", suffixes=("_1", "_2"))
        m = m.merge(sample[["pair_id", "patient_id"]], on="pair_id", how="left", suffixes=("", "_s"))
        order = {"full": 2, "partial": 1, "unsupported": 0}
        keep = m[m["consistency_1"].isin(order) & m["consistency_2"].isin(order)].copy()
        keep["a"] = keep["consistency_1"].map(order)
        keep["b"] = keep["consistency_2"].map(order)

        def stat_kappa(d):
            return kappa(d["a"].tolist(), d["b"].tolist(), 3, weighted=True)

        def stat_agree(d):
            return float((d["a"] == d["b"]).mean() * 100)

        kp, klo, khi = boot_ci(keep, stat_kappa, args.boot)
        ap_, alo, ahi = boot_ci(keep, stat_agree, args.boot)
        print("=== 4. Inter-rater agreement ===")
        print(f"  matched pairs {len(keep)}  from {keep.patient_id.nunique()} patients")
        ac = ac1(keep["a"].tolist(), keep["b"].tolist(), 3)
        pb = pabak(keep["a"].tolist(), keep["b"].tolist())
        print("  factual consistency (3 levels, linear weights)")
        print(f"    exact agreement {ap_:.1f}% ({alo:.1f}-{ahi:.1f})")
        print(f"    kappa {fmt(kp)} ({fmt(klo)}-{fmt(khi)})"
              f"   AC1 {fmt(ac)}   PABAK {fmt(pb)}")
        rows += [
            {"metric": "interrater_agreement_pct", "subset": "consistency", "value": round(ap_, 1),
             "ci_low": round(alo, 1), "ci_high": round(ahi, 1), "n": len(keep)},
            {"metric": "interrater_weighted_kappa", "subset": "consistency", "value": round(kp, 3),
             "ci_low": round(klo, 3), "ci_high": round(khi, 3), "n": len(keep)},
        ]

        # --- attributes, pooled across the five features ---
        fopt = {"ok": 2, "part": 1, "bad": 0}
        recs = []
        for _, row in m.iterrows():
            for key, _label in FEATS:
                v1, v2 = row.get(f"{key}_1", ""), row.get(f"{key}_2", "")
                if v1 in fopt and v2 in fopt:
                    recs.append({"patient_id": row["pair_id"],
                                 "a": fopt[v1], "b": fopt[v2]})
        if recs:
            attr_df = pd.DataFrame(recs)
            ka, kalo, kahi = boot_ci(
                attr_df, lambda d: kappa(d["a"].tolist(), d["b"].tolist(), 3, True), args.boot)
            aa, aalo, aahi = boot_ci(
                attr_df, lambda d: float((d["a"] == d["b"]).mean() * 100), args.boot)
            print("  clinical features (pooled over 5 attributes, linear weights)")
            print(f"    ratings {len(attr_df)} from {attr_df.patient_id.nunique()} pairs")
            acp = ac1(attr_df["a"].tolist(), attr_df["b"].tolist(), 3)
            print(f"    exact agreement {aa:.1f}% ({aalo:.1f}-{aahi:.1f})")
            print(f"    kappa {fmt(ka)} ({fmt(kalo)}-{fmt(kahi)})   AC1 {fmt(acp)}")
            rows += [
                {"metric": "interrater_attribute_agreement_pct", "subset": "features",
                 "value": round(aa, 1), "ci_low": round(aalo, 1), "ci_high": round(aahi, 1),
                 "n": len(attr_df)},
                {"metric": "interrater_attribute_kappa", "subset": "features",
                 "value": round(ka, 3), "ci_low": round(kalo, 3), "ci_high": round(kahi, 3),
                 "n": len(attr_df)},
            ]

        # --- per-attribute exact agreement, so pooled figure is not the only view ---
        print("    per attribute:")
        print(f"      {'attribute':12s} {'n':>4s} {'exact agreement':>16s}")
        for key, label in FEATS:
            sub = []
            for _, row in m.iterrows():
                v1, v2 = row.get(f"{key}_1", ""), row.get(f"{key}_2", "")
                if v1 in fopt and v2 in fopt:
                    sub.append((fopt[v1], fopt[v2]))
            if not sub:
                print(f"      {label:12s} {0:4d} {'n/a':>16s}")
                continue
            ag = sum(1 for x, y in sub if x == y) / len(sub) * 100
            kk = kappa([x for x, _ in sub], [y for _, y in sub], 3, weighted=True)
            aca = ac1([x for x, _ in sub], [y for _, y in sub], 3)
            print(f"      {label:12s} {len(sub):4d} {ag:14.1f}%   kappa {fmt(kk)}   AC1 {fmt(aca)}")
            rows.append({"metric": f"interrater_{key}_agreement_pct", "subset": "features",
                         "value": round(ag, 1), "n": len(sub)})
            rows.append({"metric": f"interrater_{key}_kappa", "subset": "features",
                         "value": round(kk, 3), "n": len(sub)})

        # --- omission type, nominal ---
        oopt = {"no": 0, "lesion": 1, "extent": 2, "assoc": 3}
        oa, ob = [], []
        for _, row in m.iterrows():
            v1, v2 = row.get("omission_1", ""), row.get("omission_2", "")
            if v1 in oopt and v2 in oopt:
                oa.append(oopt[v1])
                ob.append(oopt[v2])
        if oa:
            k_omit = kappa(oa, ob, 4, weighted=False)
            agree_omit = sum(1 for x, y in zip(oa, ob) if x == y) / len(oa) * 100
            print("  omission type (4 categories, unweighted)")
            aco = ac1(oa, ob, 4)
            print(f"    ratings {len(oa)}   exact agreement {agree_omit:.1f}%   kappa {fmt(k_omit)}   AC1 {fmt(aco)}")
            rows += [
                {"metric": "interrater_omission_agreement_pct", "subset": "omission",
                 "value": round(agree_omit, 1), "n": len(oa)},
                {"metric": "interrater_omission_kappa", "subset": "omission",
                 "value": round(k_omit, 3), "n": len(oa)},
            ]
    else:
        print("=== 4. Inter-rater agreement: skipped (only one rater supplied) ===")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print()
        print("written:", out.resolve())


if __name__ == "__main__":
    main()
