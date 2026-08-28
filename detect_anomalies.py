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
import json
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

# Accounts are 235x rarer than transactions, so the same contamination fraction
# would flag only ~22 of them. The account model uses this multiple instead.
ACCOUNT_MULTIPLIER = 4

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
        # Mean of a boolean, grouped — not .apply(lambda), which would run a
        # Python-level loop over all 4,500 groups.
        "frac_negative_balance": (trans["balance"] < 0).groupby(trans["account_id"]).mean(),
        "frac_credit": (trans["type"] == "PRIJEM").groupby(trans["account_id"]).mean(),
        "span_days": (grp["dt"].max() - grp["dt"].min()).dt.days,
    })

    # This frame is indexed by the accounts that actually appear in trans. An
    # account with no transactions would be silently absent from the model
    # rather than scored, so say so instead of quietly dropping it.
    missing = len(tables["account"]) - len(f)
    if missing:
        print(f"  warning: {missing} account(s) have no transactions and are "
              f"excluded from the account model")
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
    # `predict()` would re-traverse all 200 trees to recompute what we just
    # scored — on the 1.06M table that doubles the dominant cost. predict() is
    # defined as `score_samples < offset_`, so derive the flag from `score`
    # directly. Verified identical to predict() on both models.
    flag = score > -model.offset_
    print(f"  {label}: {len(X):,} rows x {X.shape[1]} features, "
          f"{flag.sum():,} flagged ({flag.mean():.2%}) in {time.time() - t0:.1f}s")
    return model, score, flag


# ----------------------------------------------------------------------------
# Per-record feature attribution
# ----------------------------------------------------------------------------
# Isolation Forest has no `feature_importances_` and no per-instance
# attribution: `score_samples` returns one scalar and nothing about why. So we
# reconstruct it from the isolation paths themselves.
#
# The naive approach — credit whichever feature each split used, weighted by
# depth — does not work here, and it is worth saying why. Isolation Forest picks
# its split feature *uniformly at random*, independently of the data. So the set
# of features appearing on a path is near-identical for anomalous and normal
# rows, and ranking on it returns the same three features for every record.
#
# What carries the signal is not which feature split, but how much isolation
# that split achieved. A split that drops the row from 256 candidate samples to
# 3 did the isolating; one that goes 256 -> 128 did essentially nothing. So each
# split is credited log(n_parent / n_child) along the path the row actually
# took, summed over all 200 trees and normalised to a share.
#
# This is native to the model, not a proxy: it measures the isolation the forest
# genuinely achieved per feature, rather than what merely looks unusual.

# Engineered feature -> source CSV column. Several features derive from one
# column (log_amount and amount_z both come from `amount`), so credit is
# summed per source column before ranking.
SOURCE_COLUMN = {
    "log_amount": "amount", "amount_z": "amount",
    "balance": "balance", "balance_delta": "balance", "balance_ratio": "balance",
    "days_since_prev": "date", "day_of_month": "date", "month": "date",
    "is_interbank": "bank",
}


def _to_source(name):
    """Engineered feature -> source CSV column.

    Transaction features map onto real columns of trans.csv. Account features
    (`n_txn`, `amount_std`, ...) are aggregates with no single source column, so
    they fall through unchanged. Consumers of the JSON `features` field will
    therefore see column names for trans.csv records and aggregate names for
    account.csv ones.
    """
    if name in SOURCE_COLUMN:
        return SOURCE_COLUMN[name]
    # one-hot columns are "<column>_<value>" from get_dummies
    for prefix in ("type_", "operation_", "k_symbol_"):
        if name.startswith(prefix):
            return prefix[:-1]
    return name


def _credit_matrix(model, Xv, feat_to_col, n_cols):
    """Isolation gain per source column, one row per sample."""
    n = Xv.shape[0]
    credit = np.zeros((n, n_cols))
    for est, feat_idx in zip(model.estimators_, model.estimators_features_):
        tree = est.tree_
        path = est.decision_path(Xv[:, feat_idx])       # sparse (n, n_nodes)
        nodes, indptr = path.indices, path.indptr
        rows = np.repeat(np.arange(n), np.diff(indptr))

        # Node ids increase monotonically along a root->leaf path (sklearn's
        # depth-first builder always numbers a child after its parent), and csr
        # indices come out sorted, so consecutive entries within a row are
        # exactly parent -> the child that row took.
        is_last = np.zeros(len(nodes), dtype=bool)
        is_last[indptr[1:] - 1] = True
        parent, child = nodes[~is_last], nodes[1:][~is_last[:-1]]
        row_of = rows[~is_last]

        ns = tree.n_node_samples
        gain = np.log(np.maximum(ns[parent], 1) / np.maximum(ns[child], 1))
        split_feat = tree.feature[parent]               # -2 at leaves
        internal = split_feat >= 0
        # tree.feature indexes into this estimator's feature subset, so map it
        # back through estimators_features_ to the global feature index.
        np.add.at(credit,
                  (row_of[internal], feat_to_col[feat_idx[split_feat[internal]]]),
                  gain[internal])

    # Rows have different total path gain, so normalise each to a share.
    total = credit.sum(axis=1, keepdims=True)
    return np.divide(credit, total, out=np.zeros_like(credit), where=total > 0)


def attribute_features(model, X_flagged, feature_names, top_n=3):
    """Which source columns did the forest use to isolate each row?

    Engineered features are collapsed onto their source CSV column first
    (`log_amount` and `amount_z` both count toward `amount`), so the answer is
    in terms of the dataset's own columns rather than model internals.
    """
    sources = [_to_source(f) for f in feature_names]
    uniq = sorted(set(sources))
    feat_to_col = np.array([{s: i for i, s in enumerate(uniq)}[s] for s in sources])

    credit = _credit_matrix(model, np.asarray(X_flagged, dtype="float32"),
                            feat_to_col, len(uniq))
    order = np.argsort(-credit, axis=1)[:, :top_n]
    return [[uniq[j] for j in row if credit[i, j] > 0]
            for i, row in enumerate(order)]


def normalize(score):
    """Min-max to [0,1] across the full population, so a score reads as
    'how anomalous relative to everything else', not an opaque float.

    Min-max is sensitive to a single extreme value, which rank-percentile would
    avoid. It is still the right choice here: percentile ranks compress the top
    0.5% — the only rows that reach the output — into 0.995..1.0, which is
    unreadable. Min-max keeps the flagged tail spread across ~0.7..1.0.
    Normalisation is per-model, so scores are only comparable within a dataset.
    """
    lo, hi = score.min(), score.max()
    return (score - lo) / (hi - lo) if hi > lo else np.zeros_like(score)


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

    # sklearn caps contamination at 0.5, and the account model multiplies this
    # value. Check before fitting: without it, an out-of-range value fails only
    # after the 1.06M-row transaction model has already run, with an error that
    # never mentions the multiplier.
    max_contamination = 0.5 / ACCOUNT_MULTIPLIER
    if not 0 < args.contamination <= max_contamination:
        p.error(f"--contamination must be in (0, {max_contamination}]; the account "
                f"model uses {ACCOUNT_MULTIPLIER}x it and sklearn caps that at 0.5")

    os.makedirs(args.out, exist_ok=True)
    tables = load_raw(args.raw)

    print("=== isolation forest ===")

    # --- transaction level ---
    t_keys, t_X = transaction_features(tables["trans"])
    t_model, t_score, t_flag = run_isolation_forest(t_X, args.contamination, args.seed,
                                                    label="transaction")
    txn_out = t_keys.copy()
    txn_out["anomaly_score"] = t_score
    txn_out["score_normalized"] = normalize(t_score)
    txn_out["is_anomaly"] = t_flag
    txn_out = txn_out.merge(
        tables["trans"][["trans_id", "date", "amount", "balance",
                         "type", "operation", "k_symbol"]],
        on="trans_id", how="left")
    # Feature attribution indexes t_X and txn_out by the same positional mask,
    # so the merge must not have reordered or duplicated rows. It cannot, given
    # a unique trans_id — but a duplicate key would silently misattribute every
    # record rather than fail, so assert it rather than assume it.
    assert len(txn_out) == len(t_X), "merge changed row count; trans_id not unique"

    # --- account level ---
    a_keys, a_X = account_features(tables)
    a_model, a_score, a_flag = run_isolation_forest(
        a_X, args.contamination * ACCOUNT_MULTIPLIER, args.seed, label="account    ")
    acct_out = a_keys.copy()
    acct_out["anomaly_score"] = a_score
    acct_out["score_normalized"] = normalize(a_score)
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

    # --- JSON: one object per anomaly, with the features that isolated it ---
    #
    # Records are grouped by dataset, each block sorted most-anomalous-first.
    # They are deliberately NOT sorted into one global ranking: `score` is
    # min-maxed within its own dataset, so a transaction's 1.0 and an account's
    # 1.0 are different quantities. Interleaving them would imply a comparison
    # the numbers do not support.
    records = []
    for X, out, model, ds, key in [
        (t_X, txn_out, t_model, "trans.csv", "trans_id"),
        (a_X, acct_out, a_model, "account.csv", "account_id"),
    ]:
        mask = out["is_anomaly"].values
        sub = out[mask].copy()
        # Mask before converting: np.asarray() on the full frame would
        # materialise a 220MB array to then keep 0.5% of it.
        feats = attribute_features(model, X[mask].to_numpy(), list(X.columns))
        sub["_features"] = feats
        for _, r in sub.sort_values("anomaly_score", ascending=False).iterrows():
            records.append({
                "anomaly_id": f"ANOM-{len(records) + 1:05d}",
                "dataset": ds,
                "record_id": int(r[key]),
                "score": round(float(r["score_normalized"]), 4),
                "features": r["_features"],
            })

    json_path = os.path.join(args.out, "anomalies.json")
    with open(json_path, "w") as fh:
        json.dump(records, fh, indent=2)

    print("\n=== output ===")
    print(f"  {json_path}  ({os.path.getsize(json_path) / 1e3:.0f} KB, "
          f"{len(records):,} anomalies)")
    # Row counts come from the frames we just wrote. Counting lines in the file
    # would miscount any field containing a newline, and would re-read the whole
    # 73MB table under --all-scores just to print a number.
    for path, n_rows in ((txn_csv, len(to_write)), (acct_csv, len(acct_out)),
                         (dist_csv, len(roll))):
        print(f"  {path}  ({os.path.getsize(path) / 1e3:.0f} KB, {n_rows:,} rows)")
    if not args.all_scores:
        print("  (transaction file holds flagged rows only; --all-scores for all 1.06M)")


if __name__ == "__main__":
    main()
