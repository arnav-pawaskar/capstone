"""
Great Expectations (GX 1.x) baselines on the trans table.

Install (pin and report the version):
    pip install "great_expectations==1.*" pandas numpy
The expectation class names and result_format handling below follow the GX 1.x
pandas API; verify them against the exact version you pin.

Two suites, scored exactly like the detectors in the paper:

  profiled  Expectations generated automatically from the clean baseline: PK
            uniqueness / not-null, min-max range on numeric columns, value sets for
            low-cardinality categoricals. GX 1.x ships no profiler, so this
            reproduces what its auto-suggested expectations contain. No foreign-key,
            cross-table, or group-count expectations (a profiler cannot infer them).

  hand      Expert suite: PK uniqueness, FK membership (account_id in account),
            date range, trans.date >= account.date (cross-table, needs a pandas
            merge), balance range, and a per-account-day count cap (needs a
            derived column). The two derived columns are custom pandas code outside
            GX; report this as engineering effort. Thresholds default to the clean
            baseline; override with --hand-config (JSON) and have someone who did
            NOT write the paper's detectors author it, otherwise this suite
            reintroduces the circularity.

Usage:
    python ge_baseline.py --manifest manifest.json --clean-data runs/clean \
        --out ge_results.csv [--hand-config hand.json]

hand.json keys (all optional): balance_min, balance_max, amount_min, amount_max,
date_min, date_max, burst_max.
"""
import argparse
import json
import time

import pandas as pd

import great_expectations as gx
import great_expectations.expectations as gxe

from bench_common import (load_ground_truth, load_manifest, read_table,
                          score_flags, summarize)

MAX_CATEGORICAL = 20


def make_batch(df):
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("bench_pandas")
    asset = source.add_dataframe_asset(name="trans")
    definition = asset.add_batch_definition_whole_dataframe("whole")
    return definition.get_batch(batch_parameters={"dataframe": df})


def profiled_suite(clean):
    exps = [("pk_unique", gxe.ExpectColumnValuesToBeUnique(column="trans_id")),
            ("pk_not_null", gxe.ExpectColumnValuesToNotBeNull(column="trans_id"))]
    for col in ("amount", "balance", "date"):
        exps.append((f"range_{col}", gxe.ExpectColumnValuesToBeBetween(
            column=col, min_value=float(clean[col].min()), max_value=float(clean[col].max()))))
    for col in ("type", "operation", "k_symbol", "bank"):
        vals = clean[col].dropna().unique().tolist()
        if len(vals) <= MAX_CATEGORICAL:
            exps.append((f"set_{col}", gxe.ExpectColumnValuesToBeInSet(column=col, value_set=vals)))
    return exps


def add_derived_columns(trans, account):
    df = trans.copy().reset_index(drop=True)
    acct = account.drop_duplicates("account_id")[["account_id", "date"]].rename(
        columns={"date": "account_open_date"})
    df = df.merge(acct, on="account_id", how="left")  # left join, unique right key: row order kept
    df["acct_day_count"] = df.groupby(["account_id", "date"])["trans_id"].transform("size")
    return df


def hand_suite(clean, clean_account, account, cfg):
    cd = add_derived_columns(clean, clean_account)
    c = {
        "balance_min": clean["balance"].min(), "balance_max": clean["balance"].max(),
        "date_min": clean["date"].min(), "date_max": clean["date"].max(),
        "burst_max": cd["acct_day_count"].max(),
    }
    c.update(cfg or {})
    return [
        ("pk_unique", gxe.ExpectColumnValuesToBeUnique(column="trans_id")),
        ("fk_account", gxe.ExpectColumnValuesToBeInSet(
            column="account_id", value_set=account["account_id"].dropna().unique().tolist())),
        ("date_range", gxe.ExpectColumnValuesToBeBetween(
            column="date", min_value=float(c["date_min"]), max_value=float(c["date_max"]))),
        ("date_after_open", gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="date", column_B="account_open_date", or_equal=True)),
        ("balance_range", gxe.ExpectColumnValuesToBeBetween(
            column="balance", min_value=float(c["balance_min"]), max_value=float(c["balance_max"]))),
        ("burst_cap", gxe.ExpectColumnValuesToBeBetween(
            column="acct_day_count", max_value=float(c["burst_max"]))),
    ]


def run_suite(df, suite):
    """Validate each expectation separately; return (flagged ids, per-expectation counts)."""
    batch = make_batch(df)
    ids = df["trans_id"].to_numpy()
    flagged, per_exp = set(), {}
    for label, exp in suite:
        res = batch.validate(exp, result_format="COMPLETE")
        pos = (res.result or {}).get("unexpected_index_list", [])
        hit = {int(ids[int(p)]) for p in pos}
        per_exp[label] = len(hit)
        flagged |= hit
    return flagged, per_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clean-data", required=True, help="dir with the uncorrupted tables")
    ap.add_argument("--out", default="ge_results.csv")
    ap.add_argument("--hand-config", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.hand_config)) if args.hand_config else None
    clean = read_table(args.clean_data, "trans")
    clean_account = read_table(args.clean_data, "account")

    rows = []
    for run in load_manifest(args.manifest):
        seed = run["seed"]
        trans = read_table(run["data"], "trans")
        account = read_table(run["data"], "account")
        truth = load_ground_truth(run["faults"], "trans")
        base = trans.reset_index(drop=True)
        derived = add_derived_columns(trans, account)
        suites = {"profiled": (base, profiled_suite(clean)),
                  "hand": (derived, hand_suite(clean, clean_account, account, cfg))}
        for mode, (df, suite) in suites.items():
            t0 = time.time()
            flagged, per_exp = run_suite(df, suite)
            rows.append({"seed": seed, "mode": mode, "n_expectations": len(suite),
                         "runtime_s": time.time() - t0, **score_flags(flagged, truth),
                         "per_expectation": json.dumps(per_exp)})
            print(f"seed {seed} | {mode:8s} | {len(flagged):>8,} flags | {per_exp}")

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    summ = summarize(res.drop(columns=["per_expectation"]), ["mode"])
    summ.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ[["mode", "flags_mean", "precision_lb_mean", "precision_lb_sd",
                "recall_mean", "recall_sd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
