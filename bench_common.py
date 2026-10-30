"""
Shared helpers for the PyOD and Great Expectations baseline benchmarks.

Everything is scored on the `trans` table (4,232 of the 4,299 seed-42 faults),
at record level, with the same metrics as the paper:
  * precision lower bound = |flagged ∩ injected| / |flagged|
  * recall                = |flagged ∩ injected| / |injected|   (unique record ids)
  * per-fault-type recall

Manifest (JSON) -- one entry per injection run:
[
  {"seed": 42, "data": "runs/seed_42", "faults": "runs/seed_42/injected_faults.json"},
  {"seed": 7,  "data": "runs/seed_7",  "faults": "runs/seed_7/injected_faults.json"}
]
`data` is the directory holding the *injected* copy (trans.csv, account.csv, ... ; ';' separated).

Burst ground truth: the fault log's record_id for volume_spikes is the PARENT
(account/district/disp). Detectors flag the synthetic CHILDREN, so we expand
via a `child_ids` list if the log has one, else by parsing "trans_ids A-B" from
the free-text `detail`. If neither exists the burst class is unscorable and we
raise rather than silently report 0% recall.
"""
from __future__ import annotations

import importlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FAULT_TYPES = (
    "duplicate_keys",
    "referential_breaks",
    "balance_corruption",
    "temporal_violations",
    "volume_spikes",
)


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def read_table(data_dir, name):
    return pd.read_csv(Path(data_dir) / f"{name}.csv", sep=";", low_memory=False)


def load_ground_truth(faults_path, table="trans"):
    """Return {fault_type: set(record ids)} for one table."""
    with open(faults_path) as f:
        faults = json.load(f)
    rx = re.compile(rf"{table}_ids?\s+(\d+)\s*-\s*(\d+)")
    truth = defaultdict(set)
    for fault in faults:
        if fault["table"] != table:
            continue
        ftype = fault.get("fault_type") or fault["fault_class"]  # original log uses fault_type, blind log fault_class
        if ftype == "volume_spikes":
            if "child_ids" in fault:
                ids = fault["child_ids"]
            else:
                m = rx.search(fault.get("detail", ""))
                if not m:
                    raise ValueError(
                        f"Cannot expand burst {fault.get('fault_id')}: no child_ids "
                        f"and no '{table}_ids A-B' in detail"
                    )
                ids = range(int(m.group(1)), int(m.group(2)) + 1)
            truth[ftype].update(int(i) for i in ids)
        else:
            truth[ftype].add(int(fault["record_id"]))
    return dict(truth)


def score_flags(flagged_ids, truth):
    flagged = {int(i) for i in flagged_ids}
    all_truth = set().union(*truth.values()) if truth else set()
    hit = flagged & all_truth
    out = {
        "flags": len(flagged),
        "precision_lb": len(hit) / max(len(flagged), 1),
        "recall": len(hit) / max(len(all_truth), 1),
    }
    for ftype in list(FAULT_TYPES) + sorted(t for t in truth if t not in FAULT_TYPES):
        ids = truth.get(ftype, set())
        out[f"recall_{ftype}"] = len(flagged & ids) / len(ids) if ids else np.nan
    return out


def _robust_z(s, groups):
    med = s.groupby(groups).transform("median")
    mad = (s - med).abs().groupby(groups).transform("median")
    return (s - med) / (1.4826 * mad + 1e-6)


def build_trans_features(trans, variant="full"):
    """
    Returns (X: DataFrame[float32], ids: Series of trans_id) row-aligned.

    variant:
      "full"  ~30 features: amount/balance/deltas/per-account robust z/calendar/
              account-day count/one-hot categoricals (approximates the paper's 32;
              pass --features-fn to plug in the exact matrix)
      "core6" amount, signed_amount, balance, balance_delta, balance_z, amount_z
      "z1"    balance_z only
    Raw identifiers are excluded, as in the paper.
    """
    df = trans.copy()
    for c in ("type", "operation", "k_symbol"):
        df[c] = df[c].fillna("NA").astype(str)
    df = df.sort_values(["account_id", "date", "trans_id"], kind="mergesort").reset_index(drop=True)

    sign = np.where(df["type"].str.upper() == "PRIJEM", 1.0, -1.0)
    f = pd.DataFrame(index=df.index)
    f["amount"] = df["amount"]
    f["signed_amount"] = sign * df["amount"]
    f["balance"] = df["balance"]
    f["balance_delta"] = df.groupby("account_id")["balance"].diff().fillna(0.0)
    f["balance_z"] = _robust_z(df["balance"], df["account_id"])
    f["amount_z"] = _robust_z(df["amount"], df["account_id"])

    if variant == "z1":
        return f[["balance_z"]].astype("float32"), df["trans_id"]
    if variant == "core6":
        return f.astype("float32"), df["trans_id"]

    d = df["date"].astype("int64")
    f["year"] = d // 10000
    f["month"] = (d // 100) % 100
    f["day"] = d % 100
    f["acct_day_count"] = df.groupby(["account_id", "date"])["trans_id"].transform("size")
    dt = pd.to_datetime((19000000 + d).astype(str), format="%Y%m%d", errors="coerce")
    day_ord = (dt - pd.Timestamp("1900-01-01")).dt.days
    f["days_since_prev"] = day_ord.groupby(df["account_id"]).diff().fillna(0.0)
    dummies = pd.get_dummies(df[["type", "operation", "k_symbol"]], dtype="float32")
    X = pd.concat([f.astype("float32"), dummies], axis=1)
    return X, df["trans_id"]


def get_feature_fn(spec):
    """spec = 'package.module:function'; function(trans_df, variant) -> (X, ids)."""
    if not spec:
        return build_trans_features
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def summarize(results, keys):
    metrics = [c for c in results.columns if c in ("flags", "precision_lb", "recall") or c.startswith("recall_")]
    g = results.groupby(keys)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    return out.reset_index()
