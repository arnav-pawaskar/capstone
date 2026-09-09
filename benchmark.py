"""
Multi-seed benchmark for the detector, with per-tier ablation and a
clean-baseline regression gate.

Why this exists rather than running evaluate_detection.py once:

  1. ONE injection is one sample. Precision in particular moves several points
     between seeds (measured spread: 85.8%-94.7% over five injections at
     identical settings), so a single run reports a number that is as likely to
     flatter the detector as to fault it. Every figure here is mean +/- sd.

  2. Thresholds must not be fitted to the data being scored. calibrate() reads
     them off the clean baseline, so no injection influences them, and the same
     calibration is applied unchanged to every seed. That is what makes the
     numbers comparable across seeds and defensible in a write-up.

  3. Recall alone hides where the recall comes from. The exact checks and the
     model are different instruments with wildly different precision, so the
     ablation reports each tier separately and in combination.

  4. A detector that flags clean data is broken no matter how good its recall
     is. --max-clean-flags turns that into a gate: run against an uncorrupted
     copy, and fail if the flag count regresses past a budget.

Usage:
    python benchmark.py --raw ./data/raw --work ./data/bench
    python benchmark.py --raw ./data/raw --work ./data/bench --with-model
    python benchmark.py --raw ./data/raw --work ./data/bench --max-clean-flags 150
"""

import argparse
import os
import subprocess
import sys

import numpy as np

import anomaly_detection as ad
from evaluate_detection import load_ground_truth

TIERS = ("structural", "constraint", "model")


def ensure_injection(raw, work, seed, rate):
    """Generate the fault-injected copy for `seed` unless it is already there."""
    out = os.path.join(work, f"seed{seed}")
    if os.path.exists(os.path.join(out, "injected_faults.json")):
        return out
    print(f"  injecting seed {seed} ...", flush=True)
    subprocess.run([sys.executable, "inject_faults2.py", "--raw", raw,
                    "--out", out, "--seed", str(seed), "--rate", str(rate)],
                   check=True, stdout=subprocess.DEVNULL)
    return out


def detect(tables, cal, with_model, contamination, seed, n_estimators, day_chain=True):
    """-> {tier: {(table, record_id)}}"""
    out = {t: set() for t in TIERS}
    for records in (ad.structural_checks(tables),
                    ad.constraint_checks(tables, cal, day_chain=day_chain)):
        for r in records:
            out[r["tier"]].add((r["dataset"].removesuffix(".csv"), r["record_id"]))
    if with_model:
        for table in ad.TABLES:
            ids, X, names = ad.FEATURE_BUILDERS[table](tables)
            X = X.fillna(0.0).astype(float)
            if len(X) < 10:
                continue
            _, _, flag, _ = ad.run_scoped_forest(X, names, contamination, seed, n_estimators)
            ids = np.asarray(ids)
            out["model"] |= {(table, int(r)) for r in ids[flag]}
    return out


def score(flagged, gt):
    tp = len(flagged & gt)
    p = tp / max(len(flagged), 1)
    r = tp / max(len(gt), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), len(flagged)


def summarise(label, rows):
    """rows: list of (precision, recall, f1, n_flags) across seeds."""
    a = np.array([r[:3] for r in rows])
    flags = np.mean([r[3] for r in rows])
    return (f"{label:<34}{flags:>9,.0f}"
            f"{a[:, 0].mean():>8.1%} +/-{a[:, 0].std():<6.3f}"
            f"{a[:, 1].mean():>8.1%} +/-{a[:, 1].std():<6.3f}"
            f"{a[:, 2].mean():>8.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="./data/raw",
                   help="clean baseline: thresholds are calibrated on it, faults injected into it")
    p.add_argument("--work", default="./data/bench", help="where injected copies are cached")
    p.add_argument("--seeds", default="42,7,13,99,2024")
    p.add_argument("--rate", type=float, default=0.001)
    p.add_argument("--with-model", action="store_true",
                   help="also run the Isolation Forest tier (slow: one scoped ensemble "
                        "per table, minutes per seed on the 1.06M-row transaction table)")
    p.add_argument("--contamination", type=float, default=0.005)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--seed", type=int, default=42, help="model seed, not the injection seed")
    p.add_argument("--max-clean-flags", type=int, default=None,
                   help="fail with exit code 1 if the review tier flags more than this "
                        "many records in the CLEAN baseline. Use as a regression gate.")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.work, exist_ok=True)

    print("=== calibrating on clean baseline ===")
    clean = ad.load_raw(args.raw)
    cal = ad.calibrate(clean)
    print(f"  bursts {cal['burst']}   years {cal['year']['trans']}")
    print(f"  balance robust-z: strong {cal['balance_z']:.2f}, weak {cal['balance_z_weak']:.2f}")

    print("\n=== clean-baseline regression check ===")
    clean_tiers = detect(clean, cal, args.with_model, args.contamination,
                         args.seed, args.n_estimators)
    clean_review = clean_tiers["structural"] | clean_tiers["constraint"]
    print(f"  review-tier flags on uncorrupted data: {len(clean_review):,}  "
          f"(structural {len(clean_tiers['structural']):,}, "
          f"constraint {len(clean_tiers['constraint']):,})")
    if args.with_model:
        print(f"  model-tier flags on uncorrupted data : {len(clean_tiers['model']):,}"
              f"   <- fires on a fixed quota regardless of whether anything is wrong")

    print(f"\n=== detecting across {len(seeds)} injections ===")
    per_tier = {t: [] for t in TIERS}
    review, everything = [], []
    for s in seeds:
        path = ensure_injection(args.raw, args.work, s, args.rate)
        tables = ad.load_raw(path)
        gt = set(load_ground_truth(os.path.join(path, "injected_faults.json")))
        tiers = detect(tables, cal, args.with_model, args.contamination,
                       args.seed, args.n_estimators)
        for t in TIERS:
            per_tier[t].append(score(tiers[t], gt))
        rv = tiers["structural"] | tiers["constraint"]
        review.append(score(rv, gt))
        everything.append(score(rv | tiers["model"], gt))
        print(f"  seed {s:<6} review: precision {review[-1][0]:6.1%}  "
              f"recall {review[-1][1]:6.1%}  F1 {review[-1][2]:6.1%}")

    print(f"\n=== ablation over {len(seeds)} injections (mean +/- sd) ===")
    print(f"{'tier':<34}{'flags':>9}{'precision':>19}{'recall':>19}{'F1':>8}")
    print("-" * 89)
    print(summarise("structural only", per_tier["structural"]))
    print(summarise("constraint only", per_tier["constraint"]))
    print(summarise("REVIEW (structural + constraint)", review))
    if args.with_model:
        print(summarise("model only", per_tier["model"]))
        print(summarise("all tiers merged", everything))

    if args.max_clean_flags is not None and len(clean_review) > args.max_clean_flags:
        print(f"\nFAIL: {len(clean_review):,} review-tier flags on clean data exceeds "
              f"--max-clean-flags={args.max_clean_flags:,}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
