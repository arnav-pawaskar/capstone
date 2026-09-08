"""
Fault injection for the Berka / PKDD'99 banking dataset.

Berka ships no fraud labels, which is exactly why validate() in detect_anomalies.py
has to triangulate instead of measuring precision/recall directly. This script
gives you the ground truth that's otherwise missing: it injects five controlled
fault classes into copies of the raw CSVs and logs exactly what it changed, so you
can run anomaly_detection.py against the output and check how many injected faults
it actually caught. (Use anomaly_detection.py, not detect_anomalies.py: the latter
asserts trans_id is unique, which duplicate_keys deliberately violates.)

Each fault type is applied to every table where it's semantically meaningful, not
just trans/account — see the *_SPECS constants below for exactly which
table+column combinations each fault type touches, and why:

  duplicate_keys       every table: same primary key gets a second, conflicting row
  referential_breaks   every child table: an FK is rewritten to a nonexistent parent
                        (district has no FK of its own, so it never appears here —
                        that's correct, not a gap)
  balance_corruption   only tables with a real money field: trans.balance,
                        loan.amount, order.amount
  temporal_violations  only tables with a real date field: trans.date, loan.date,
                        card.issued (all checked against account open date), plus
                        account.date checked against its own first transaction
  volume_spikes        trans (many transactions crammed onto one day for one
                        account), loan (many loans granted the same day across one
                        district), card (many cards issued the same day to one disp)

Each fault is logged with enough detail (table, key, old value, new value) to check
later whether your detector's flagged trans_id/account_id set actually covers it.

Usage:
    python inject_faults2.py --raw ./data/raw --out ./data/faulty
    python inject_faults2.py --raw ./data/raw --out ./data/faulty --rate 0.002 --seed 7
    python inject_faults2.py --raw ./data/raw --out ./data/faulty \\
        --only duplicate_keys,volume_spikes

Then:
    python anomaly_detection.py --raw ./data/faulty --out ./artifacts_faulty
    # compare anomalies.json / anomaly_scores_*.csv against injected_faults.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]

# district.csv ships with opaque A1..A16 headers in the original distribution.
# A1 is district_id, A4 is an arbitrary numeric demographic field used only for
# duplicate-key perturbation below. If your copy already has descriptive headers,
# change these two.
DISTRICT_ID_COL = "A1"
DISTRICT_NUMERIC_COL = "A4"

CHILD_PK = {"trans": "trans_id", "disp": "disp_id", "card": "card_id",
            "loan": "loan_id", "order": "order_id", "account": "account_id",
            "client": "client_id", "district": DISTRICT_ID_COL}

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

# (table, primary key col, column to perturb, perturb kind)
# perturb kind: "days" (shift a YYMMDD date), "mult" (multiply a numeric value),
# "add" (add a random integer offset), "choice" (swap to a different category)
DUPLICATE_SPECS = [
    ("account", "account_id", "date", "days"),
    ("client", "client_id", "birth_number", "add"),
    ("disp", "disp_id", "type", "choice", ["OWNER", "DISPONENT"]),
    ("card", "card_id", "type", "choice", ["classic", "junior", "gold"]),
    ("district", DISTRICT_ID_COL, DISTRICT_NUMERIC_COL, "mult"),
    ("loan", "loan_id", "amount", "mult"),
    ("order", "order_id", "amount", "mult"),
    ("trans", "trans_id", "amount", "mult"),
]

# (table, primary key col, money column)
BALANCE_SPECS = [
    ("trans", "trans_id", "balance"),
    ("loan", "loan_id", "amount"),
    ("order", "order_id", "amount"),
]


def load_raw(raw_dir):
    return {name: pd.read_csv(os.path.join(raw_dir, f"{name}.csv"), sep=";", low_memory=False)
            for name in TABLES}


def parse_yymmdd(x):
    return pd.to_datetime(str(int(x)).zfill(6), format="%y%m%d")


def format_yymmdd(dt):
    return int(dt.strftime("%y%m%d"))


def parse_flex_date(x):
    """Parse a YYMMDD value that may be a plain int, or a string like
    '970124 00:00:00' — card.csv's 'issued' column ships that way in some
    redistributions of this dataset, unlike every other date column here."""
    s = str(x).strip().split(" ")[0].split(".")[0]
    return pd.to_datetime(s.zfill(6), format="%y%m%d")


def format_flex_date(dt, like):
    """Format dt back in the same style as `like`: plain YYMMDD int, or a
    'YYMMDD 00:00:00' string, whichever the original value looked like."""
    base = dt.strftime("%y%m%d")
    if isinstance(like, str) and " " in like:
        return f"{base} 00:00:00"
    return int(base)


# ----------------------------------------------------------------------------
def inject_duplicate_keys(tables, rng, rate, log):
    """Append a second row under an existing primary key, with a conflicting
    field so it's a real inconsistency, not just a harmless clone. This is
    also a direct test of any code downstream that assumes the key is unique
    (e.g. detect_anomalies.py asserts trans_id is unique after a merge)."""
    for spec in DUPLICATE_SPECS:
        table, pk, col, kind = spec[0], spec[1], spec[2], spec[3]
        df = tables[table]
        if col not in df.columns:
            print(f"  warning: {table}.{col} not found, skipping duplicate_keys on {table}")
            continue
        n = max(1, round(rate * len(df)))
        idx = rng.choice(df.index, size=min(n, len(df)), replace=False)
        dup_rows = df.loc[idx].copy()

        if kind == "days":
            shift = rng.integers(1, 90, size=len(dup_rows))
            dup_rows[col] = [format_yymmdd(parse_yymmdd(d) + pd.Timedelta(days=int(s)))
                              for d, s in zip(dup_rows[col], shift)]
        elif kind == "mult":
            dup_rows[col] = (dup_rows[col] * rng.uniform(0.5, 2.0, size=len(dup_rows))).round(1)
        elif kind == "add":
            dup_rows[col] = dup_rows[col] + rng.integers(-500, 500, size=len(dup_rows))
        elif kind == "choice":
            choices = spec[4]
            dup_rows[col] = rng.choice(choices, size=len(dup_rows))

        tables[table] = pd.concat([df, dup_rows], ignore_index=True)
        for _, r in dup_rows.iterrows():
            log.append({"fault_type": "duplicate_keys", "table": table,
                         "record_id": int(r[pk]),
                         "detail": f"duplicated {pk} with {col} changed to {r[col]}"})


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
    """Shock a money field to an implausible value, or swap it with another
    row's, so it looks locally wrong relative to that record's own history.
    Applied to every table with a genuine money field: trans.balance,
    loan.amount, order.amount."""
    for table, pk, col in BALANCE_SPECS:
        df = tables[table]
        # The shock below produces a float (factor + gaussian noise) and the
        # swap can move a float into an int column. loan.amount ships as int64,
        # and pandas 3.x refuses that point-write outright rather than upcasting.
        # Widen once, up front, so the fault types are order-independent --
        # otherwise this only works when duplicate_keys happens to run first.
        df[col] = df[col].astype(float)
        n = max(1, round(rate * len(df)))
        pool = list(rng.permutation(df.index))[:n]
        while pool:
            if rng.random() < 0.7 or len(pool) < 2:
                i = pool.pop()
                old = df.at[i, col]
                factor = rng.choice([-1, 8, 15, -8])
                new = round(float(old) * factor + rng.normal(0, 50), 1)
                df.at[i, col] = new
                log.append({"fault_type": "balance_corruption", "table": table,
                             "record_id": int(df.at[i, pk]), "mode": "shock",
                             "detail": f"{col} {old} -> {new}"})
            else:
                i, j = pool.pop(), pool.pop()
                old_i, old_j = df.at[i, col], df.at[j, col]
                df.at[i, col], df.at[j, col] = old_j, old_i
                log.append({"fault_type": "balance_corruption", "table": table,
                             "record_id": int(df.at[i, pk]), "mode": "swap",
                             "detail": f"{col} swapped with {pk} {int(df.at[j, pk])}"})
                log.append({"fault_type": "balance_corruption", "table": table,
                             "record_id": int(df.at[j, pk]), "mode": "swap",
                             "detail": f"{col} swapped with {pk} {int(df.at[i, pk])}"})


def _temporal_break(df, date_col, pk_col, table_name, ref_lookup, ref_desc,
                     max_date_dt, rate, rng, log):
    """Move date_col before some reference date (account open date, typically),
    or beyond the dataset's date range entirely. Sorting can't repair either —
    they're wrong in an absolute sense, not just out of order."""
    n = max(1, round(rate * len(df)))
    idx = rng.choice(df.index, size=min(n, len(df)), replace=False)
    for i in idx:
        old_date = df.at[i, date_col]
        ref_val = ref_lookup(df.loc[i])
        if ref_val is not None and rng.random() < 0.6:
            new_dt = parse_yymmdd(ref_val) - pd.Timedelta(days=int(rng.integers(30, 500)))
            submode = f"before_{ref_desc}"
        else:
            new_dt = max_date_dt + pd.Timedelta(days=int(rng.integers(100, 600)))
            submode = "beyond_dataset_range"
        new_date = format_flex_date(new_dt, old_date)
        df.at[i, date_col] = new_date
        log.append({"fault_type": "temporal_violations", "table": table_name,
                     "record_id": int(df.at[i, pk_col]), "mode": submode,
                     "detail": f"{date_col} {old_date} -> {new_date}"})


def inject_temporal_violations(tables, rng, rate, log):
    trans, account, loan, card, disp = (tables["trans"], tables["account"],
                                         tables["loan"], tables["card"], tables["disp"])
    acct_open = dict(zip(account["account_id"], account["date"]))
    max_date_dt = parse_yymmdd(trans["date"].max())

    # trans.date should never precede its own account's open date
    _temporal_break(trans, "date", "trans_id", "trans",
                     lambda row: acct_open.get(int(row["account_id"])),
                     "account_open", max_date_dt, rate, rng, log)

    # loan.date should never precede its own account's open date
    _temporal_break(loan, "date", "loan_id", "loan",
                     lambda row: acct_open.get(int(row["account_id"])),
                     "account_open", max_date_dt, rate, rng, log)

    # card.issued should never precede the open date of the account behind its disp
    disp_acct = dict(zip(disp["disp_id"], disp["account_id"]))

    def card_ref(row):
        acc = disp_acct.get(int(row["disp_id"]))
        return acct_open.get(acc) if acc is not None else None

    _temporal_break(card, "issued", "card_id", "card", card_ref,
                     "account_open", max_date_dt, rate, rng, log)

    # account.date should never come AFTER that account's own first transaction
    first_trans_date = trans.groupby("account_id")["date"].min().to_dict()
    n = max(1, round(rate * len(account)))
    idx = rng.choice(account.index, size=min(n, len(account)), replace=False)
    for i in idx:
        acc_id = int(account.at[i, "account_id"])
        first_dt_raw = first_trans_date.get(acc_id)
        if first_dt_raw is None:
            continue
        old_date = account.at[i, "date"]
        new_dt = parse_yymmdd(first_dt_raw) + pd.Timedelta(days=int(rng.integers(10, 200)))
        new_date = format_yymmdd(new_dt)
        account.at[i, "date"] = new_date
        log.append({"fault_type": "temporal_violations", "table": "account",
                     "record_id": acc_id, "mode": "opened_after_first_transaction",
                     "detail": f"date {old_date} -> {new_date}"})


def _spike_trans(tables, rng, n_accounts, spike_size, log):
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

        log.append({"fault_type": "volume_spikes", "table": "trans",
                     "record_id": int(acc), "mode": "burst",
                     "detail": f"{spike_size} synthetic transactions added on date {spike_date} "
                               f"(that day normally had {len(day_rows)}); "
                               f"trans_ids {new_ids[0]}-{new_ids[-1]}"})

    if new_rows:
        new_df = pd.DataFrame(new_rows).reindex(columns=trans.columns)
        tables["trans"] = pd.concat([trans, new_df], ignore_index=True)


def _spike_loan(tables, rng, n_bursts, log):
    """Grant a burst of synthetic loans, all on the same day, to accounts in
    one district. Also breaks the real-world 'at most one loan per account'
    business rule, which is itself a realistic symptom of a duplicate loan-
    origination bug."""
    account, loan = tables["account"], tables["loan"]
    if "district_id" not in account.columns or len(loan) == 0:
        return
    districts = account["district_id"].unique()
    chosen = rng.choice(districts, size=min(n_bursts, len(districts)), replace=False)
    next_loan_id = int(loan["loan_id"].max()) + 1
    new_rows = []
    burst_size = 5

    for dist in chosen:
        candidates = account.loc[account["district_id"] == dist, "account_id"].values
        if len(candidates) < burst_size:
            continue
        accs = rng.choice(candidates, size=burst_size, replace=False)
        burst_date = int(rng.choice(loan["date"].values))
        hist_amount, hist_duration, hist_payments, hist_status = (
            loan["amount"].values, loan["duration"].values,
            loan["payments"].values, loan["status"].values)
        new_ids = []
        for acc in accs:
            row = {c: np.nan for c in loan.columns}
            row.update({
                "loan_id": next_loan_id, "account_id": int(acc), "date": burst_date,
                "amount": float(rng.choice(hist_amount)),
                "duration": int(rng.choice(hist_duration)),
                "payments": float(rng.choice(hist_payments)),
                "status": rng.choice(hist_status),
            })
            new_rows.append(row)
            new_ids.append(next_loan_id)
            next_loan_id += 1
        log.append({"fault_type": "volume_spikes", "table": "loan",
                     "record_id": int(dist), "mode": "burst",
                     "detail": f"{burst_size} synthetic loans granted on date {burst_date} "
                               f"across district {int(dist)}; loan_ids {new_ids[0]}-{new_ids[-1]}"})

    if new_rows:
        new_df = pd.DataFrame(new_rows).reindex(columns=loan.columns)
        tables["loan"] = pd.concat([loan, new_df], ignore_index=True)


def _spike_card(tables, rng, n_bursts, log):
    """Issue a burst of synthetic cards, all on the same day, to one
    disposition — several cards suddenly appearing on one account is a
    realistic card-fraud / issuance-bug signature."""
    disp, card = tables["disp"], tables["card"]
    if len(card) == 0:
        return
    disp_ids = disp["disp_id"].unique()
    chosen = rng.choice(disp_ids, size=min(n_bursts, len(disp_ids)), replace=False)
    next_card_id = int(card["card_id"].max()) + 1
    new_rows = []
    burst_size = 5
    hist_type = card["type"].dropna().values
    hist_issued = card["issued"].dropna().values

    for d in chosen:
        sample_like = rng.choice(hist_issued)
        burst_date = format_flex_date(parse_flex_date(sample_like), sample_like)
        new_ids = []
        for _ in range(burst_size):
            row = {c: np.nan for c in card.columns}
            row.update({
                "card_id": next_card_id, "disp_id": int(d),
                "type": rng.choice(hist_type) if len(hist_type) else np.nan,
                "issued": burst_date,
            })
            new_rows.append(row)
            new_ids.append(next_card_id)
            next_card_id += 1
        log.append({"fault_type": "volume_spikes", "table": "card",
                     "record_id": int(d), "mode": "burst",
                     "detail": f"{burst_size} synthetic cards issued on date {burst_date} "
                               f"to disp_id {int(d)}; card_ids {new_ids[0]}-{new_ids[-1]}"})

    if new_rows:
        new_df = pd.DataFrame(new_rows).reindex(columns=card.columns)
        tables["card"] = pd.concat([card, new_df], ignore_index=True)


def inject_volume_spikes(tables, rng, n_accounts, spike_size, log):
    _spike_trans(tables, rng, n_accounts, spike_size, log)
    _spike_loan(tables, rng, max(1, n_accounts // 3), log)
    _spike_card(tables, rng, max(1, n_accounts // 3), log)


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
                        "referential/balance/temporal), applied independently per table")
    p.add_argument("--spike-accounts", type=int, default=5,
                   help="number of trans burst locations; loan/card bursts scale down from this")
    p.add_argument("--spike-size", type=int, default=30,
                   help="synthetic transactions per spiked account (loan/card bursts are fixed at 5)")
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

    print("\n=== faults by table ===")
    print(summary.groupby(["fault_type", "table"]).size().unstack(fill_value=0))

    print("\n=== output ===")
    for name in TABLES:
        delta = len(tables[name]) - before_counts[name]
        note = f" (+{delta} rows)" if delta else ""
        print(f"  {os.path.join(args.out, name + '.csv')}: {len(tables[name]):,} rows{note}")
    print(f"  {os.path.join(args.out, 'injected_faults.json')}: {len(log)} faults")
    print(f"  {os.path.join(args.out, 'injected_faults.csv')}: flat ground-truth table")
    print("\nNext: python anomaly_detection.py --raw", args.out, "--out ./artifacts_faulty")


if __name__ == "__main__":
    main()
