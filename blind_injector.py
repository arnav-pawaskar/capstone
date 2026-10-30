"""
Detector-blind fault injector for the PKDD'99 tables.

PURPOSE: produce faults that none of the paper's detectors were written against,
to test the circularity limitation. The faults below were designed from the
SCHEMA ONLY and deliberately avoid what the three tiers check (duplicate PKs,
dangling FKs, dates before account open / outside the year range, per-parent
same-day bursts, extreme or swapped balances).

BLINDNESS PROTOCOL (see the write-up): the claim is only valid if
  1. the detectors are frozen first (git tag; pass --detector-commit),
  2. the person who edits --spec / picks the seed and runs this script has not
     read the detector code, and
  3. blind_faults.json (--faults-out) is kept away from the detector authors
     until the frozen detectors have produced their output.

Injection is done on a CLEAN copy, not on the already-injected copy, so blind
faults never overlap the original five classes and the detectors' behaviour on
the clean data is the reference point.

Usage:
    python blind_injector.py --clean-data data/clean --out-dir runs/blind \
        --faults-out sealed/blind_faults.json --spec blind_spec.json \
        --detector-commit <git sha>

Outputs: out-dir/<all 8 tables> (trans.csv and loan.csv rewritten, others copied
byte-for-byte), out-dir/blind_manifest.json (spec, counts, sha256 of outputs),
and the ground-truth log at --faults-out.

Fault classes (all new, none targeted by a check):
  categorical_mislabel   trans.operation swapped to another VALID category
  unit_scale_window      a contiguous run of one account's amounts x1000 or /1000
  zero_amount            amount replaced by 0 (default-value imputation)
  replayed_txn           an existing transaction re-inserted under a NEW trans_id
  sign_flip              type PRIJEM<->VYDAJ with amount unchanged
  date_shift_inrange     date moved +/- weeks, staying valid and after account open
  balance_offset_window  a contiguous run of balances shifted by a moderate constant
  loan_payment_mismatch  loan.payments inconsistent with amount/duration
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ["account", "client", "disp", "district", "card", "loan", "order", "trans"]

DEFAULT_SPEC = {
    "seed": 20260101,
    "classes": {
        "categorical_mislabel": {"enabled": True, "rate": 0.001},
        "unit_scale_window": {"enabled": True, "rate": 0.001, "window": [5, 15], "factors": [1000.0, 0.001]},
        "zero_amount": {"enabled": True, "rate": 0.001},
        "replayed_txn": {"enabled": True, "rate": 0.001},
        "sign_flip": {"enabled": True, "rate": 0.001},
        "date_shift_inrange": {"enabled": True, "rate": 0.001, "days": [30, 120]},
        "balance_offset_window": {"enabled": True, "rate": 0.001, "window": [5, 15], "magnitude": [0.1, 0.5]},
        "loan_payment_mismatch": {"enabled": True, "rate": 0.02, "min_count": 8, "factor": [1.5, 3.0]},
    },
}


# ----------------------------------------------------------------- helpers
def _py(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _deep_merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _to_dt(s):
    return pd.to_datetime((19000000 + s.astype("int64")).astype(str), format="%Y%m%d", errors="coerce")


def _to_int(ts):
    return int((ts.year - 1900) * 10000 + ts.month * 100 + ts.day)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ctx:
    def __init__(self, trans, account, loan, rng):
        self.trans, self.account, self.loan, self.rng = trans, account, loan, rng
        self.orig = trans.copy()
        self.used = np.zeros(len(trans), dtype=bool)
        self.faults, self.extra_rows = [], []
        order = trans.sort_values(["account_id", "date", "trans_id"], kind="mergesort").index.to_numpy()
        accts = trans.loc[order, "account_id"].to_numpy()
        self.order = {a: p.to_numpy() for a, p in pd.Series(order).groupby(accts)}

    def log(self, cls, table, rid, mode, old, new, **extra):
        self.faults.append({
            "fault_id": f"BLIND-{len(self.faults) + 1:05d}", "fault_class": cls, "table": table,
            "record_id": _py(rid), "mode": mode, "old": _py(old), "new": _py(new), **extra})


def quota(cfg, n):
    return max(int(cfg.get("min_count", 0)), int(round(cfg["rate"] * n)))


def pick_rows(ctx, k, eligible=None):
    ok = ~ctx.used if eligible is None else (~ctx.used & eligible)
    avail = np.flatnonzero(ok)
    pos = ctx.rng.choice(avail, size=min(k, len(avail)), replace=False)
    ctx.used[pos] = True
    return pos


def pick_windows(ctx, total_rows, lo, hi):
    accts = [a for a, p in ctx.order.items() if len(p) >= hi + 2]
    out, covered, tries = [], 0, 0
    while covered < total_rows and tries < 100 * max(total_rows, 1):
        tries += 1
        p = ctx.order[accts[int(ctx.rng.integers(len(accts)))]]
        length = int(ctx.rng.integers(lo, hi + 1))
        start = int(ctx.rng.integers(0, len(p) - length + 1))
        win = p[start:start + length]
        if ctx.used[win].any():
            continue
        ctx.used[win] = True
        out.append((len(out), win))
        covered += length
    return out


# ----------------------------------------------------------- fault classes
def categorical_mislabel(ctx, cfg):
    col, t = "operation", ctx.trans
    vals = t[col].dropna().unique().tolist()
    for p in pick_rows(ctx, quota(cfg, len(t)), t[col].notna().to_numpy()):
        old = t.at[p, col]
        choices = [v for v in vals if v != old]
        new = choices[int(ctx.rng.integers(len(choices)))]
        t.at[p, col] = new
        ctx.log("categorical_mislabel", "trans", t.at[p, "trans_id"], col, old, new)


def unit_scale_window(ctx, cfg):
    t = ctx.trans
    for wid, win in pick_windows(ctx, quota(cfg, len(t)), *cfg["window"]):
        f = float(ctx.rng.choice(cfg["factors"]))
        for p in win:
            old = t.at[p, "amount"]
            t.at[p, "amount"] = round(float(old) * f, 1)
            ctx.log("unit_scale_window", "trans", t.at[p, "trans_id"], f"x{f:g}", old,
                    t.at[p, "amount"], window_id=wid)


def zero_amount(ctx, cfg):
    t = ctx.trans
    for p in pick_rows(ctx, quota(cfg, len(t)), (t["amount"] > 0).to_numpy()):
        old = t.at[p, "amount"]
        t.at[p, "amount"] = 0.0
        ctx.log("zero_amount", "trans", t.at[p, "trans_id"], "default_zero", old, 0.0)


def replayed_txn(ctx, cfg):
    o = ctx.orig
    k = quota(cfg, len(o))
    src = ctx.rng.choice(len(o), size=k, replace=False)
    next_id = int(max(o["trans_id"].max(), ctx.trans["trans_id"].max())) + 1
    for i, p in enumerate(src):
        row = o.iloc[p].to_dict()
        new_id = next_id + i
        row["trans_id"] = new_id
        ctx.extra_rows.append(row)
        ctx.log("replayed_txn", "trans", new_id, "copy_new_key", None, None,
                source_trans_id=int(o.iloc[p]["trans_id"]), alt_ids=[int(o.iloc[p]["trans_id"])])


def sign_flip(ctx, cfg):
    t = ctx.trans
    flip = {"PRIJEM": "VYDAJ", "VYDAJ": "PRIJEM", "VYBER": "PRIJEM"}
    for p in pick_rows(ctx, quota(cfg, len(t)), t["type"].isin(flip).to_numpy()):
        old = t.at[p, "type"]
        t.at[p, "type"] = flip[old]
        ctx.log("sign_flip", "trans", t.at[p, "trans_id"], "type", old, flip[old])


def date_shift_inrange(ctx, cfg):
    t = ctx.trans
    dt = _to_dt(t["date"])
    open_dt = t["account_id"].map(dict(zip(ctx.account["account_id"], _to_dt(ctx.account["date"]))))
    y_lo, y_hi = dt.min().year, dt.max().year  # valid range taken from the (clean) data
    ok = (dt.notna() & open_dt.notna()).to_numpy()
    lo, hi = cfg["days"]
    for p in pick_rows(ctx, quota(cfg, len(t)), ok):
        delta = int(ctx.rng.integers(lo, hi + 1))
        sign = 1 if ctx.rng.random() < 0.5 else -1
        for s in (sign, -sign):
            new = dt.iat[p] + pd.Timedelta(days=s * delta)
            if new >= open_dt.iat[p] and y_lo <= new.year <= y_hi:
                old = t.at[p, "date"]
                t.at[p, "date"] = _to_int(new)
                ctx.log("date_shift_inrange", "trans", t.at[p, "trans_id"], f"{s * delta:+d}d", old, _to_int(new))
                break
        else:
            ctx.used[p] = False  # no valid shift for this row; skip it


def balance_offset_window(ctx, cfg):
    t = ctx.trans
    std = ctx.orig.groupby("account_id")["balance"].std().fillna(0.0)
    lo_m, hi_m = cfg["magnitude"]
    for wid, win in pick_windows(ctx, quota(cfg, len(t)), *cfg["window"]):
        acct = t.at[win[0], "account_id"]
        mag = float(ctx.rng.uniform(lo_m, hi_m)) * max(float(std.get(acct, 0.0)), 100.0)
        off = round(mag * (1 if ctx.rng.random() < 0.5 else -1), 1)
        for p in win:
            old = t.at[p, "balance"]
            t.at[p, "balance"] = round(float(old) + off, 1)
            ctx.log("balance_offset_window", "trans", t.at[p, "trans_id"], f"{off:+.1f}", old,
                    t.at[p, "balance"], window_id=wid)


def loan_payment_mismatch(ctx, cfg):
    loan = ctx.loan
    k = quota(cfg, len(loan))
    for p in ctx.rng.choice(len(loan), size=k, replace=False):
        f = float(ctx.rng.uniform(*cfg["factor"]))
        f = f if ctx.rng.random() < 0.5 else 1.0 / f
        old = loan.at[p, "payments"]
        loan.at[p, "payments"] = round(float(old) * f, 1)
        ctx.log("loan_payment_mismatch", "loan", loan.at[p, "loan_id"], f"x{f:.2f}", old, loan.at[p, "payments"])


CLASSES = {
    "categorical_mislabel": categorical_mislabel,
    "unit_scale_window": unit_scale_window,
    "zero_amount": zero_amount,
    "replayed_txn": replayed_txn,
    "sign_flip": sign_flip,
    "date_shift_inrange": date_shift_inrange,
    "balance_offset_window": balance_offset_window,
    "loan_payment_mismatch": loan_payment_mismatch,
}


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--faults-out", required=True, help="ground-truth log; keep it away from detector authors")
    ap.add_argument("--spec", default=None, help="JSON overriding DEFAULT_SPEC (seed, classes, rates)")
    ap.add_argument("--detector-commit", default="UNSPECIFIED")
    args = ap.parse_args()

    spec = copy.deepcopy(DEFAULT_SPEC)
    if args.spec:
        _deep_merge(spec, json.load(open(args.spec)))

    src, out = Path(args.clean_data), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shutil.copy2(src / f"{name}.csv", out / f"{name}.csv")

    read = lambda n: pd.read_csv(src / f"{n}.csv", sep=";", low_memory=False)
    trans, account, loan = read("trans"), read("account"), read("loan")
    ctx = Ctx(trans, account, loan, np.random.default_rng(spec["seed"]))

    for name, fn in CLASSES.items():
        cfg = spec["classes"].get(name, {})
        if cfg.get("enabled", False):
            before = len(ctx.faults)
            fn(ctx, cfg)
            print(f"{name:24s} {len(ctx.faults) - before:6d} faulted rows")

    if ctx.extra_rows:
        trans = pd.concat([ctx.trans, pd.DataFrame(ctx.extra_rows)[list(ctx.trans.columns)]], ignore_index=True)
    else:
        trans = ctx.trans
    trans.to_csv(out / "trans.csv", sep=";", index=False)
    ctx.loan.to_csv(out / "loan.csv", sep=";", index=False)

    faults_path = Path(args.faults_out)
    faults_path.parent.mkdir(parents=True, exist_ok=True)
    with open(faults_path, "w") as f:
        json.dump(ctx.faults, f, indent=1)

    counts = pd.Series([x["fault_class"] for x in ctx.faults]).value_counts().to_dict()
    manifest = {
        "spec": spec, "detector_commit": args.detector_commit, "fault_counts": counts,
        "total_faulted_rows": len(ctx.faults), "pandas": pd.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0],
        "sha256": {n: _sha256(out / f"{n}.csv") for n in TABLES},
        "faults_log_sha256": _sha256(faults_path),
    }
    with open(out / "blind_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{len(ctx.faults)} faulted rows -> {faults_path}\nmanifest -> {out / 'blind_manifest.json'}")


if __name__ == "__main__":
    main()
