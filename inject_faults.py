"""
Fault injection for the Berka / PKDD'99 banking dataset.

Berka ships no fraud labels, which is exactly why validate() in detect_anomalies.py
has to triangulate instead of measuring precision/recall directly. This script
gives you the ground truth that's otherwise missing: it injects five controlled
fault classes into copies of the raw CSVs and logs exactly what it changed, so you
can run detect_anomalies.py against the output and check how many injected faults
it actually caught.

Fault classes:
  duplicate_keys       existing primary key gets a second, conflicting row
  referential_breaks   a foreign key is rewritten to point at a nonexistent parent
  balance_corruption   a transaction's balance is shocked or swapped with another's
  temporal_violations  a transaction's date is moved before its account existed,
                        or beyond the dataset's date range entirely
  volume_spikes        a burst of synthetic transactions crammed onto one day for
                        one account, otherwise plausible in type/amount

Each fault is logged with enough detail (table, key, old value, new value) to check
later whether your detector's flagged trans_id/account_id set actually covers it.

Usage:
    python inject_faults.py --raw ./data/raw --out ./data/faulty
    python inject_faults.py --raw ./data/raw --out ./data/faulty --rate 0.002 --seed 7
    python inject_faults.py --raw ./data/raw --out ./data/faulty \\
        --only duplicate_keys,volume_spikes

Then:
    python detect_anomalies.py --raw ./data/faulty --out ./artifacts_faulty
    # compare anomalies.json / anomaly_scores_*.csv against injected_faults.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]

# district.csv ships with opaque A1..A16 headers in the original distribution.
# A1 is district_id. If your copy already has descriptive headers, change this.
DISTRICT_ID_COL = "A1"

CHILD_PK = {"trans": "trans_id", "disp": "disp_id", "card": "card_id",
            "loan": "loan_id", "order": "order_id", "account": "account_id",
            "client": "client_id"}

# (child table, fk column, parent table, parent key column)
FK_SPECS = [
    ("trans", "account_id", "account", "account_id"),
    ("disp", "account_id", "account", "account_id"),
    ("disp", "client_id", "client", "client_id"),
    ("card", "disp_id", "disp", "disp_id"),
    ("loan", "account_id", "account", "account_id"),
    ("order", "account_id", "account", "account_id"),
    ("account", "district_id", "district", DISTRICT_ID_COL),
    ("client", "district_id", "district", DISTRICT_ID_COL),
]


def load_raw(raw_dir):
    return {name: pd.read_csv(os.path.join(raw_dir, f"{name}.csv"), sep=";", low_memory=False)
            for name in TABLES}


def parse_yymmdd(x):
    return pd.to_datetime(str(int(x)).zfill(6), format="%y%m%d")


def format_yymmdd(dt):
    return int(dt.strftime("%y%m%d"))


# ----------------------------------------------------------------------------
def inject_duplicate_keys(tables, rng, rate, log):
    """Append a second row under an existing primary key, with a conflicting
    field so it's a real inconsistency, not just a harmless clone. This is
    also a direct test of any code downstream that assumes the key is unique
    (e.g. detect_anomalies.py asserts trans_id is unique after a merge)."""
    # trans: same trans_id, perturbed amount, unperturbed balance -> the two
    # rows disagree about what actually happened.
    trans = tables["trans"]
    n = max(1, round(rate * len(trans)))
    idx = rng.choice(trans.index, size=min(n, len(trans)), replace=False)
    dup_rows = trans.loc[idx].copy()
    dup_rows["amount"] = (dup_rows["amount"] * rng.uniform(0.5, 2.0, size=len(dup_rows))).round(1)
    tables["trans"] = pd.concat([trans, dup_rows], ignore_index=True)
    for _, r in dup_rows.iterrows():
        log.append({"fault_type": "duplicate_keys", "table": "trans",
                     "record_id": int(r["trans_id"]),
                     "detail": f"duplicated trans_id with amount changed to {r['amount']}"})

    # account: same account_id, creation date shifted by a few weeks.
    account = tables["account"]
    n2 = max(1, round(rate * len(account) * 0.3))  # accounts are rarer; a smaller absolute count
    idx2 = rng.choice(account.index, size=min(n2, len(account)), replace=False)
    dup_acct = account.loc[idx2].copy()
    shift = rng.integers(1, 90, size=len(dup_acct))
    dup_acct["date"] = [format_yymmdd(parse_yymmdd(d) + pd.Timedelta(days=int(s)))
                         for d, s in zip(dup_acct["date"], shift)]
    tables["account"] = pd.concat([account, dup_acct], ignore_index=True)
    for _, r in dup_acct.iterrows():
        log.append({"fault_type": "duplicate_keys", "table": "account",
                     "record_id": int(r["account_id"]),
                     "detail": f"duplicated account_id with date changed to {r['date']}"})


def inject_referential_breaks(tables, rng, rate, log):
    """Rewrite a foreign key to a value guaranteed absent from its parent table."""
    for child, fk_col, parent, parent_key in FK_SPECS:
        parent_df = tables[parent]
        if parent_key not in parent_df.columns:
            print(f"  warning: {parent}.{parent_key} not found, skipping FK check "
                  f"{child}.{fk_col} -> {parent}.{parent_key}")
            continue
        child_df = tables[child]
        n = max(1, round(rate * len(child_df)))
        idx = rng.choice(child_df.index, size=min(n, len(child_df)), replace=False)
        parent_max = int(parent_df[parent_key].max())
        for offset, i in enumerate(idx):
            old_val = child_df.at[i, fk_col]
            new_val = parent_max + 1000 + offset
            child_df.at[i, fk_col] = new_val
            log.append({"fault_type": "referential_breaks", "table": child,
                         "record_id": int(child_df.at[i, CHILD_PK[child]]),
                         "detail": f"{fk_col} {old_val} -> {new_val} (absent from {parent})"})


def inject_balance_corruption(tables, rng, rate, log):
    """Shock a balance to an implausible value, or swap two transactions'
    balances so each looks locally wrong relative to its own account's history."""
    trans = tables["trans"]
    n = max(1, round(rate * len(trans)))
    pool = list(rng.permutation(trans.index))[:n]
    while pool:
        if rng.random() < 0.7 or len(pool) < 2:
            i = pool.pop()
            old = trans.at[i, "balance"]
            factor = rng.choice([-1, 8, 15, -8])
            new = round(float(old) * factor + rng.normal(0, 50), 1)
            trans.at[i, "balance"] = new
            log.append({"fault_type": "balance_corruption", "table": "trans",
                         "record_id": int(trans.at[i, "trans_id"]), "mode": "shock",
                         "detail": f"balance {old} -> {new}"})
        else:
            i, j = pool.pop(), pool.pop()
            old_i, old_j = trans.at[i, "balance"], trans.at[j, "balance"]
            trans.at[i, "balance"], trans.at[j, "balance"] = old_j, old_i
            log.append({"fault_type": "balance_corruption", "table": "trans",
                         "record_id": int(trans.at[i, "trans_id"]), "mode": "swap",
                         "detail": f"balance swapped with trans_id {int(trans.at[j, 'trans_id'])}"})
            log.append({"fault_type": "balance_corruption", "table": "trans",
                         "record_id": int(trans.at[j, "trans_id"]), "mode": "swap",
                         "detail": f"balance swapped with trans_id {int(trans.at[i, 'trans_id'])}"})


def inject_temporal_violations(tables, rng, rate, log):
    """Move a transaction's date before its own account existed, or beyond the
    dataset's date range. Sorting can't repair either of these — they're wrong
    in an absolute sense, not just out of order."""
    trans = tables["trans"]
    acct_open = dict(zip(tables["account"]["account_id"], tables["account"]["date"]))
    max_date_dt = parse_yymmdd(trans["date"].max())

    n = max(1, round(rate * len(trans)))
    idx = rng.choice(trans.index, size=min(n, len(trans)), replace=False)
    for i in idx:
        account_id = int(trans.at[i, "account_id"])
        old_date = trans.at[i, "date"]
        if rng.random() < 0.6 and account_id in acct_open:
            creation_dt = parse_yymmdd(acct_open[account_id])
            new_dt = creation_dt - pd.Timedelta(days=int(rng.integers(30, 500)))
            submode = "pre_account_open"
        else:
            new_dt = max_date_dt + pd.Timedelta(days=int(rng.integers(100, 600)))
            submode = "beyond_dataset_range"
        new_date = format_yymmdd(new_dt)
        trans.at[i, "date"] = new_date
        log.append({"fault_type": "temporal_violations", "table": "trans",
                     "record_id": int(trans.at[i, "trans_id"]), "mode": submode,
                     "detail": f"date {old_date} -> {new_date} (account_id {account_id})"})


def inject_volume_spikes(tables, rng, n_accounts, spike_size, log):
    """Crowd spike_size synthetic transactions onto one existing day for one
    account, sampling type/operation/k_symbol/amount from that account's own
    history so frequency — not magnitude — is the only anomalous signal."""
    trans = tables["trans"]
    acct_ids = trans["account_id"].unique()
    chosen = rng.choice(acct_ids, size=min(n_accounts, len(acct_ids)), replace=False)
    next_trans_id = int(trans["trans_id"].max()) + 1
    new_rows = []

    for acc in chosen:
        sub = trans[trans["account_id"] == acc].sort_values(["date", "trans_id"])
        if len(sub) < 5:
            continue  # too little history to sample a plausible spike from
        spike_date = int(rng.choice(sub["date"].unique()))
        day_rows = sub[sub["date"] == spike_date]
        running_balance = float(day_rows["balance"].iloc[-1])
        hist_types = sub["type"].dropna().values
        hist_ops = sub["operation"].dropna().values
        hist_ksym = sub["k_symbol"].dropna().values
        hist_amounts = sub["amount"].values

        new_ids = []
        for _ in range(spike_size):
            t = rng.choice(hist_types)
            amt = round(float(rng.choice(hist_amounts)) * rng.uniform(0.8, 1.2), 1)
            running_balance += amt if t == "PRIJEM" else -amt
            row = {c: np.nan for c in trans.columns}
            row.update({
                "trans_id": next_trans_id, "account_id": int(acc), "date": spike_date,
                "type": t,
                "operation": rng.choice(hist_ops) if len(hist_ops) else np.nan,
                "amount": amt, "balance": round(running_balance, 1),
                "k_symbol": rng.choice(hist_ksym) if len(hist_ksym) and rng.random() < 0.5 else np.nan,
            })
            new_rows.append(row)
            new_ids.append(next_trans_id)
            next_trans_id += 1

        normal_daily_count = len(day_rows)
        log.append({"fault_type": "volume_spikes", "table": "trans",
                     "record_id": int(acc), "mode": "burst",
                     "detail": f"{spike_size} synthetic transactions added on date {spike_date} "
                               f"(that day normally had {normal_daily_count}); "
                               f"trans_ids {new_ids[0]}-{new_ids[-1]}"})

    if new_rows:
        new_df = pd.DataFrame(new_rows).reindex(columns=trans.columns)
        tables["trans"] = pd.concat([trans, new_df], ignore_index=True)


FAULT_REGISTRY = {
    "duplicate_keys": lambda tables, rng, args, log: inject_duplicate_keys(tables, rng, args.rate, log),
    "referential_breaks": lambda tables, rng, args, log: inject_referential_breaks(tables, rng, args.rate, log),
    "balance_corruption": lambda tables, rng, args, log: inject_balance_corruption(tables, rng, args.rate, log),
    "temporal_violations": lambda tables, rng, args, log: inject_temporal_violations(tables, rng, args.rate, log),
    "volume_spikes": lambda tables, rng, args, log: inject_volume_spikes(
        tables, rng, args.spike_accounts, args.spike_size, log),
}
FAULT_ORDER = ["duplicate_keys", "referential_breaks", "balance_corruption",
               "temporal_violations", "volume_spikes"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="./data/raw")
    p.add_argument("--out", default="./data/faulty")
    p.add_argument("--rate", type=float, default=0.001,
                   help="fraction of rows to corrupt per point-fault type (duplicate/"
                        "referential/balance/temporal); each is computed independently")
    p.add_argument("--spike-accounts", type=int, default=5,
                   help="number of accounts to inject a volume spike into")
    p.add_argument("--spike-size", type=int, default=30,
                   help="synthetic transactions per spiked account")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", default=None,
                   help="comma-separated subset of fault types to run, e.g. "
                        "'duplicate_keys,volume_spikes'. Default: all.")
    args = p.parse_args()

    selected = FAULT_ORDER if args.only is None else [s.strip() for s in args.only.split(",")]
    unknown = set(selected) - set(FAULT_REGISTRY)
    if unknown:
        p.error(f"unknown fault type(s): {unknown}. choose from {list(FAULT_REGISTRY)}")

    rng = np.random.default_rng(args.seed)
    tables = load_raw(args.raw)
    before_counts = {name: len(df) for name, df in tables.items()}

    print("=== fault injection ===")
    log = []
    for name in FAULT_ORDER:
        if name not in selected:
            continue
        t0 = len(log)
        FAULT_REGISTRY[name](tables, rng, args, log)
        print(f"  {name}: {len(log) - t0} faults logged")

    os.makedirs(args.out, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(args.out, f"{name}.csv"), sep=";", index=False)

    for i, fault in enumerate(log):
        fault["fault_id"] = f"FAULT-{i + 1:05d}"
    with open(os.path.join(args.out, "injected_faults.json"), "w") as fh:
        json.dump(log, fh, indent=2, default=str)

    summary = pd.DataFrame(log)[["fault_id", "fault_type", "table", "record_id", "detail"]]
    summary.to_csv(os.path.join(args.out, "injected_faults.csv"), index=False)

    print("\n=== output ===")
    for name in TABLES:
        delta = len(tables[name]) - before_counts[name]
        note = f" (+{delta} rows)" if delta else ""
        print(f"  {os.path.join(args.out, name + '.csv')}: {len(tables[name]):,} rows{note}")
    print(f"  {os.path.join(args.out, 'injected_faults.json')}: {len(log)} faults")
    print(f"  {os.path.join(args.out, 'injected_faults.csv')}: flat ground-truth table")
    print("\nNext: python detect_anomalies.py --raw", args.out, "--out ./artifacts_faulty")


if __name__ == "__main__":
    main()
