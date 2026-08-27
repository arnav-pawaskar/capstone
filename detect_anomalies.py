"""
Unsupervised anomaly detection over the Berka / PKDD'99 banking dataset.

Isolation Forest scores every row on how few random axis-aligned splits it takes
to isolate. Rows that fall out early are, by construction, unlike the bulk of the
data — no rule written in advance, no labels required.

Two models, two granularities:

  TRANSACTION  ~1.06M rows -> which individual movements look wrong
  ACCOUNT      4.5K rows   -> which whole accounts behave unlike their peers

Standalone: reads the eight raw CSVs, writes CSVs. No database, no graph engine.
Only pandas / numpy / scikit-learn.

Nothing here is supervised. Berka ships no fraud labels, so `--contamination` is
a budget you choose, not a measurement. Loan status is deliberately held OUT of
the feature matrix and used afterwards as an independent check (see `validate`).

Usage:
    python detect_anomalies.py --raw ./data/raw --out ./artifacts
    python detect_anomalies.py --contamination 0.01        # wider net
    python detect_anomalies.py --all-scores                # dump all 1.06M scores
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# district.csv ships with opaque A1..A16 headers.
DISTRICT_COLS = [
    "district_id", "name", "region", "pop", "m499", "m1999", "m9999", "m10000",
    "n_cities", "urban_ratio", "avg_salary", "unemp95", "unemp96",
    "entrep_per1000", "crime95", "crime96",
]

TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]

# Berka loan status: A finished/paid, B finished/unpaid, C running/ok, D running/debt.
# B and D are the bank's own "this went wrong" marker.
BAD_LOAN_STATUS = {"B", "D"}


def load_raw(raw_dir):
    """Read the eight semicolon-delimited CSVs into DataFrames."""
    tables = {}
    for name in TABLES:
        df = pd.read_csv(os.path.join(raw_dir, f"{name}.csv"), sep=";", low_memory=False)
        if name == "district":
            df.columns = DISTRICT_COLS
        tables[name] = df
    return tables


def parse_berka_date(s):
    """YYMMDD int -> datetime. The dataset spans 1993-1998, so no century ambiguity."""
    return pd.to_datetime(s.astype(str).str.zfill(6), format="%y%m%d")


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def transaction_features(trans):
    """One row per transaction. Mix of raw, per-account-relative, and temporal.

    The per-account-relative features are what make this worth doing: a 50K
    withdrawal is unremarkable globally and very remarkable on an account whose
    every other movement is under 2K. Isolation Forest only sees columns, so
    that context has to be computed into one.
    """
    df = trans.copy()
    df["dt"] = parse_berka_date(df["date"])
    df = df.sort_values(["account_id", "dt", "trans_id"]).reset_index(drop=True)
    grp = df.groupby("account_id", sort=False)

    feats = pd.DataFrame(index=df.index)

    # --- raw magnitude. log1p because amount is heavily right-skewed. ---
    feats["log_amount"] = np.log1p(df["amount"])
    feats["balance"] = df["balance"]

    # --- movement relative to the account's own history ---
    prev_balance = grp["balance"].shift(1)
    feats["balance_delta"] = (df["balance"] - prev_balance).fillna(0.0)

    acct_mean = grp["amount"].transform("mean")
    acct_std = grp["amount"].transform("std").fillna(0.0)
    # An account with one transaction, or N identical ones, has std 0 -> the
    # z-score is undefined, not infinite. Zero is the honest encoding.
    z = (df["amount"] - acct_mean) / acct_std.replace(0, np.nan)
    feats["amount_z"] = z.fillna(0.0)

    # Balance as a fraction of the account's own ceiling: catches an account
    # draining to near-zero even when the absolute numbers look ordinary.
    acct_max_bal = grp["balance"].transform("max")
    feats["balance_ratio"] = np.where(acct_max_bal > 0,
                                      df["balance"] / acct_max_bal, 0.0)

    # --- temporal ---
    feats["days_since_prev"] = grp["dt"].diff().dt.days.fillna(-1.0)  # -1 = first txn
    feats["day_of_month"] = df["dt"].dt.day
    feats["month"] = df["dt"].dt.month

    # --- categorical, one-hot. Cardinality is 3/6/9, so this stays narrow. ---
    cats = pd.DataFrame({
        "type": df["type"].fillna("NONE"),
        "operation": df["operation"].fillna("NONE"),
        # k_symbol carries both NaN and a literal blank string; same meaning.
        "k_symbol": df["k_symbol"].fillna("NONE").replace(r"^\s*$", "NONE", regex=True),
    })
    feats = pd.concat([feats, pd.get_dummies(cats, dtype=float)], axis=1)

    # Interbank transfers name a counterparty bank; intrabank ones don't.
    feats["is_interbank"] = df["bank"].notna().astype(float)

    keys = df[["trans_id", "account_id"]].copy()
    return keys, feats.astype("float64")


def account_features(tables):
    """One row per account: how does this account behave over its whole life?

    Loan status is deliberately excluded — it is the validation signal, and
    feeding it in would make the check circular.
    """
    trans = tables["trans"].copy()
    trans["dt"] = parse_berka_date(trans["date"])
    grp = trans.groupby("account_id")

    f = pd.DataFrame({
        "n_txn": grp.size(),
        "amount_mean": grp["amount"].mean(),
        "amount_std": grp["amount"].std().fillna(0.0),
        "amount_max": grp["amount"].max(),
        "balance_mean": grp["balance"].mean(),
        "balance_min": grp["balance"].min(),
        "balance_max": grp["balance"].max(),
        "balance_std": grp["balance"].std().fillna(0.0),
        "frac_negative_balance": grp["balance"].apply(lambda s: (s < 0).mean()),
        "frac_credit": grp["type"].apply(lambda s: (s == "PRIJEM").mean()),
        "span_days": (grp["dt"].max() - grp["dt"].min()).dt.days,
    })
    f["txn_per_day"] = f["n_txn"] / f["span_days"].clip(lower=1)

    # Product holdings — an account with 3 cards and no loan is a different
    # animal from one with a loan and no card.
    for name in ("order", "loan"):
        f[f"n_{name}"] = tables[name].groupby("account_id").size().reindex(f.index).fillna(0.0)
    # Cards hang off disp, which hangs off account.
    card_acct = tables["card"].merge(tables["disp"][["disp_id", "account_id"]],
                                     on="disp_id", how="left")
    f["n_card"] = card_acct.groupby("account_id").size().reindex(f.index).fillna(0.0)
    f["n_client"] = tables["disp"].groupby("account_id").size().reindex(f.index).fillna(0.0)

    keys = pd.DataFrame({"account_id": f.index}).reset_index(drop=True)
    return keys, f.reset_index(drop=True).astype("float64")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def run_isolation_forest(X, contamination, seed=42, label=""):
    """Fit and score. No scaling: Isolation Forest splits one feature at a time,
    so it is invariant to per-feature monotone rescaling."""
    t0 = time.time()
    model = IsolationForest(
        n_estimators=200,
        max_samples=256,       # the value the original paper recommends
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    ).fit(X)
    # score_samples: higher = more normal. Negate so higher = more anomalous,
    # which is the direction every downstream query wants.
    score = -model.score_samples(X)
    flag = model.predict(X) == -1
    print(f"  {label}: {len(X):,} rows x {X.shape[1]} features, "
          f"{flag.sum():,} flagged ({flag.mean():.2%}) in {time.time() - t0:.1f}s")
    return score, flag


# ----------------------------------------------------------------------------
# Validation — no labels exist, so triangulate
# ----------------------------------------------------------------------------
def validate(txn_out, acct_out, tables):
    print("\n=== validation ===")

    # 1. Against a hand-written rule (balance < 0). Agreement says the model
    #    rediscovered what a rule already knew; the gap is what it adds.
    neg = txn_out["balance"] < 0
    flagged = txn_out["is_anomaly"]
    both = int((neg & flagged).sum())
    print(f"  negative-balance rows       : {int(neg.sum()):,}")
    print(f"  of those, also IF-flagged   : {both:,} ({both / max(neg.sum(), 1):.1%})")
    print(f"  IF-flagged but NOT negative : {int((flagged & ~neg).sum()):,} "
          f"<- rows no balance rule catches")

    # 2. Against loan status, held out of the feature matrix. If the account
    #    model learned anything real, defaulted accounts should be
    #    over-represented among its flags.
    loans = tables["loan"][["account_id", "status"]].copy()
    loans["bad"] = loans["status"].isin(BAD_LOAN_STATUS)
    merged = acct_out.merge(loans, on="account_id", how="inner")
    base = merged["bad"].mean()
    among = merged.loc[merged["is_anomaly"], "bad"]
    print(f"\n  accounts with a loan        : {len(merged):,}")
    print(f"  base bad-loan rate          : {base:.2%}")
    if len(among) and base > 0:
        print(f"  bad-loan rate among flagged : {among.mean():.2%} (n={len(among)})"
              f"  -> lift {among.mean() / base:.2f}x")

    print("\n  top 5 most anomalous transactions:")
    print(txn_out.nlargest(5, "anomaly_score")[
        ["trans_id", "account_id", "amount", "balance", "type", "operation",
         "anomaly_score"]].to_string(index=False))


def district_rollup(txn_out, tables):
    """Where do the anomalies cluster geographically?

    Raw counts just rank districts by size — the biggest district always wins,
    which is not a finding. Normalise by each district's own transaction volume
    so the number means "unusually anomalous", not "large".
    """
    acct_dist = tables["account"][["account_id", "district_id"]]
    dist = tables["district"][["district_id", "name", "region"]]
    j = (txn_out[["trans_id", "account_id", "is_anomaly"]]
         .merge(acct_dist, on="account_id", how="left")
         .merge(dist, on="district_id", how="left"))

    roll = j.groupby(["name", "region"], as_index=False).agg(
        anomalies=("is_anomaly", "sum"), total_txns=("is_anomaly", "size"))
    roll["anomaly_rate"] = roll["anomalies"] / roll["total_txns"]
    baseline = roll["anomalies"].sum() / roll["total_txns"].sum()
    roll["lift"] = roll["anomaly_rate"] / baseline
    roll = roll.rename(columns={"name": "district"})

    print(f"\n=== anomalies by district (baseline rate {baseline:.3%}) ===")
    print("  by rate, districts with >= 2000 transactions:")
    print(roll[roll["total_txns"] >= 2000].nlargest(10, "anomaly_rate")
          .to_string(index=False))
    print("\n  by raw count (size-confounded, shown for contrast):")
    print(roll.nlargest(5, "anomalies")[
        ["district", "anomalies", "total_txns", "anomaly_rate"]].to_string(index=False))
    return roll.sort_values("anomaly_rate", ascending=False)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="./data/raw")
    p.add_argument("--out", default="./artifacts")
    p.add_argument("--contamination", type=float, default=0.005,
                   help="expected anomaly fraction; a budget you choose, not a measurement")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-scores", action="store_true",
                   help="write scores for all 1.06M transactions (~73MB) instead of "
                        "just the flagged ones")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tables = load_raw(args.raw)

    print("=== isolation forest ===")

    # --- transaction level ---
    t_keys, t_X = transaction_features(tables["trans"])
    t_score, t_flag = run_isolation_forest(t_X, args.contamination, args.seed,
                                           label="transaction")
    txn_out = t_keys.copy()
    txn_out["anomaly_score"] = t_score
    txn_out["is_anomaly"] = t_flag
    txn_out = txn_out.merge(
        tables["trans"][["trans_id", "date", "amount", "balance",
                         "type", "operation", "k_symbol"]],
        on="trans_id", how="left")

    # --- account level ---
    # Accounts are 235x rarer than transactions, so the same fraction would flag
    # only ~22 of them. 4x gives a set big enough to say anything about.
    a_keys, a_X = account_features(tables)
    a_score, a_flag = run_isolation_forest(a_X, args.contamination * 4, args.seed,
                                           label="account    ")
    acct_out = a_keys.copy()
    acct_out["anomaly_score"] = a_score
    acct_out["is_anomaly"] = a_flag

    validate(txn_out, acct_out, tables)
    roll = district_rollup(txn_out, tables)

    # --- write ---
    # Default to flagged rows only: the full 1.06M-row score table is ~73MB,
    # which does not belong in a git repo. --all-scores overrides for tuning.
    txn_csv = os.path.join(args.out, "anomaly_scores_trans.csv")
    to_write = txn_out if args.all_scores else txn_out[txn_out["is_anomaly"]]
    to_write.sort_values("anomaly_score", ascending=False).to_csv(txn_csv, index=False)

    acct_csv = os.path.join(args.out, "anomaly_scores_account.csv")
    acct_out.sort_values("anomaly_score", ascending=False).to_csv(acct_csv, index=False)

    dist_csv = os.path.join(args.out, "anomaly_by_district.csv")
    roll.to_csv(dist_csv, index=False)

    print(f"\n=== output ===")
    for path in (txn_csv, acct_csv, dist_csv):
        print(f"  {path}  ({os.path.getsize(path) / 1e3:.0f} KB, "
              f"{sum(1 for _ in open(path)) - 1:,} rows)")
    if not args.all_scores:
        print("  (transaction file holds flagged rows only; --all-scores for all 1.06M)")


if __name__ == "__main__":
    main()
