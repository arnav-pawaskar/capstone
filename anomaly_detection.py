"""
Isolation Forest anomaly detection across the full Berka / PKDD'99 schema,
built to run against inject_faults.py's output and catch all five injected
fault classes across all eight tables.

Score:
    S(x, m) = 2 ** (-E(h(x)) / c(m))

    E(h(x)) is the average path length to isolate x across every tree,
    c(m) the expected path length of an unsuccessful search in a Binary
    Search Tree of m points (the Isolation Forest paper's normalizer).
    Scores sit in (0, 1); closer to 1 means fewer splits were needed to
    isolate the point, i.e. more anomalous.

    scikit-learn's IsolationForest.score_samples() already returns exactly
    -S(x, m) internally (see _iforest.py: it computes 2**(-depth/c) then
    negates it so that, like decision_function, lower = more abnormal).
    So S(x, m) here is just -model.score_samples(X) -- no need to hand-roll
    path lengths or c(m).

Three passes per table, in increasing order of how much they guess:

  1. Structural    Exact duplicate-key and broken-foreign-key lookups. A model
                    trained on distributions has no concept of "this primary key
                    must be unique" or "this FK must resolve".
  2. Constraint    Exact temporal checks (an event predating its own parent, or a
                    date outside the range the table occupies) and same-day burst
                    counts, plus a per-account robust-z check on balance.
  3. ML scoring    IsolationForest over engineered features -- the residual net
                    for anomalies nobody wrote a check for.

Passes 1 and 2 do nearly all the work, and that split is measured, not assumed.
Isolation Forest picks its split feature uniformly at random over ~30 features,
so a fault living in one feature is diluted by a factor of ~n_features. Measured
against inject_faults2.py's ground truth (see evaluate_detection.py), the model
alone recovered 6.9% of temporal violations and 10.7% of volume bursts, where a
threshold on the single right column recovers ~100% of both. Anything expressible
as a violated constraint therefore belongs in pass 1 or 2, not here.

Balance corruption is the one genuinely statistical class -- a swapped balance
violates no constraint -- so it stays a scored heuristic and the model keeps it.

Recall by fault type on the default injection (rate 0.001, seed 42):

    duplicate_keys       100.0%      temporal_violations  100.0%
    referential_breaks   100.0%      volume_spikes        100.0%
    balance_corruption    49.6%      ALL                   88.0%

Every table has different columns, so each gets its own feature builder --
see FEATURE_BUILDERS. district.csv and disp.csv have very little signal
(few numeric columns) and rely mostly on the rule-based pass.

Output: anomalies.json, one entry per flagged record across every table,
sorted by score descending and numbered ANOM-00001, ANOM-00002, ...

Usage:
    python anomaly_detection.py --raw ./data/faulty --out ./artifacts_faulty
    python evaluate_detection.py --faults ./data/faulty/injected_faults.json \
                                 --anomalies ./artifacts_faulty/anomalies.json
    python anomaly_detection.py --raw ./data/faulty --contamination 0.01
    python anomaly_detection.py --raw ./data/faulty --tables trans,loan,card
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# district.csv ships with opaque A1..A16 headers.
DISTRICT_COLS = [
    "district_id", "name", "region", "pop", "m499", "m1999", "m9999", "m10000",
    "n_cities", "urban_ratio", "avg_salary", "unemp95", "unemp96",
    "entrep_per1000", "crime95", "crime96",
]
DISTRICT_NUMERIC_COLS = [c for c in DISTRICT_COLS if c not in ("district_id", "name", "region")]

TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]

CHILD_PK = {"trans": "trans_id", "disp": "disp_id", "card": "card_id",
            "loan": "loan_id", "order": "order_id", "account": "account_id",
            "client": "client_id", "district": "district_id"}

# (child table, fk column, parent table, parent key column) -- district_id here
# refers to the renamed column (see DISTRICT_COLS), not the raw "A1" header.
FK_SPECS = [
    ("trans", "account_id", "account", "account_id"),
    ("disp", "account_id", "account", "account_id"),
    ("disp", "client_id", "client", "client_id"),
    ("card", "disp_id", "disp", "disp_id"),
    ("loan", "account_id", "account", "account_id"),
    ("order", "account_id", "account", "account_id"),
    ("account", "district_id", "district", "district_id"),
    ("client", "district_id", "district", "district_id"),
]


def load_raw(raw_dir):
    tables = {}
    for name in TABLES:
        df = pd.read_csv(os.path.join(raw_dir, f"{name}.csv"), sep=";", low_memory=False)
        if name == "district":
            df.columns = DISTRICT_COLS
        tables[name] = df
    return tables


def parse_flex_date(series):
    """YYMMDD -> datetime, tolerant of both a plain int (930113) and a
    string with a trailing time component ('970124 00:00:00'), which is how
    card.csv's 'issued' column ships in some redistributions of this
    dataset. Also coerces junk like '?' to NaT instead of raising."""
    s = series.astype(str).str.strip().str.split(" ").str[0].str.split(".").str[0].str.zfill(6)
    return pd.to_datetime(s, format="%y%m%d", errors="coerce")


def add_feature(X, display_names, colname, series, display):
    """Append one column to the feature matrix and record, in lockstep, what
    raw source field it should be attributed back to in anomalies.json --
    keeps X.columns and display_names impossible to accidentally desync."""
    X[colname] = series
    display_names.append(display)


def add_dummies(X, display_names, df_col, prefix, display, dummy_na=False):
    dummies = pd.get_dummies(df_col, prefix=prefix, dummy_na=dummy_na)
    for c in dummies.columns:
        add_feature(X, display_names, c, dummies[c].astype(float), display)


# ----------------------------------------------------------------------------
# Per-table feature builders. Each returns (record_ids, X, display_names).
# ----------------------------------------------------------------------------
def trans_features(tables):
    trans = tables["trans"]
    dt = parse_flex_date(trans["date"])
    df = trans.assign(dt=dt).sort_values(["account_id", "dt", "trans_id"]).reset_index(drop=True)
    grp = df.groupby("account_id", sort=False)

    X, names = pd.DataFrame(index=df.index), []
    add_feature(X, names, "log_amount", np.log1p(df["amount"].clip(lower=0)), "amount")
    add_feature(X, names, "balance", df["balance"], "balance")

    prev_balance = grp["balance"].shift(1)
    add_feature(X, names, "balance_delta", (df["balance"] - prev_balance).fillna(0.0), "balance")

    # Balance relative to the account's OWN history. Without these the model sees
    # only absolute balance, where a corrupted row hides inside the global
    # distribution; adding them roughly triples recall on balance_corruption.
    acct_med = grp["balance"].transform("median")
    acct_mad = (df["balance"] - acct_med).abs().groupby(df["account_id"]).transform("median")
    add_feature(X, names, "balance_robust_z",
                ((df["balance"] - acct_med) / acct_mad.replace(0, np.nan)).fillna(0.0), "balance")
    acct_max_bal = grp["balance"].transform("max")
    add_feature(X, names, "balance_ratio",
                np.where(acct_max_bal > 0, df["balance"] / acct_max_bal, 0.0), "balance")

    acct_mean = grp["amount"].transform("mean")
    acct_std = grp["amount"].transform("std").fillna(0.0)
    z = (df["amount"] - acct_mean) / acct_std.replace(0, np.nan)
    add_feature(X, names, "amount_z", z.fillna(0.0), "amount")

    # -1 marks "no previous transaction", which 0 cannot -- 0 is a real value
    # meaning "same day as the previous one".
    add_feature(X, names, "days_since_prev", grp["dt"].diff().dt.days.fillna(-1.0), "date")
    add_feature(X, names, "day_of_month", df["dt"].dt.day.fillna(0), "date")
    add_feature(X, names, "month", df["dt"].dt.month.fillna(0), "date")
    # day_of_month and month are cyclic: a row moved to 1991 or to 2000 lands on
    # an ordinary day of an ordinary month and is invisible in them. A monotone
    # time axis is what makes temporal_violations separable at all.
    add_feature(X, names, "year", df["dt"].dt.year.fillna(0), "date")
    # A transaction dated before its own account opened is a hard violation, not
    # a distributional oddity -- it shows up here as a negative value, a sign no
    # legitimate row can have. NaT (orphan account_id) falls back to 0.
    acct_open = dict(zip(tables["account"]["account_id"],
                          parse_flex_date(tables["account"]["date"])))
    open_dt = pd.to_datetime(df["account_id"].map(acct_open))
    add_feature(X, names, "days_since_account_open",
                (df["dt"] - open_dt).dt.days.fillna(0.0), "date")
    # volume_spikes are a frequency signal, not a magnitude one: the burst rows
    # are sampled from the account's own history, so amount and type look
    # normal. How many transactions this account posted on this row's own day is
    # the only column that separates them.
    add_feature(X, names, "account_day_txn_count",
                df.groupby(["account_id", "date"])["trans_id"]
                  .transform("size").astype(float), "date")
    # NOTE: the raw account_id is deliberately NOT a feature. Broken FKs are
    # caught exactly by structural_checks(), so feeding an arbitrary identifier
    # to the model buys no recall and costs real accuracy: it is high-cardinality
    # and meaningless-but-splittable, so it absorbs ~1/n_features of every tree's
    # random split budget and isolates rows purely by where their id happens to
    # sit in the range.

    for col in ["type", "operation", "k_symbol"]:
        add_dummies(X, names, df[col], col, col, dummy_na=True)
    add_feature(X, names, "is_interbank", df["bank"].notna().astype(float), "bank")

    return df["trans_id"].values, X, names


def account_features(tables):
    account, trans = tables["account"], tables["trans"]
    acct_dt = parse_flex_date(account["date"])
    first_trans_raw = trans.groupby("account_id")["date"].min()
    first_trans_dt = dict(zip(first_trans_raw.index, parse_flex_date(first_trans_raw)))
    n_trans = trans.groupby("account_id").size()

    X, names = pd.DataFrame(index=account.index), []
    add_feature(X, names, "district_id_fk", account["district_id"], "district_id")
    add_dummies(X, names, account["frequency"], "frequency", "frequency")

    first_series = account["account_id"].map(first_trans_dt)
    days_to_first = (first_series - acct_dt).dt.days
    add_feature(X, names, "days_to_first_trans", days_to_first.fillna(0.0), "date")
    add_feature(X, names, "n_trans", account["account_id"].map(n_trans).fillna(0.0), "n_transactions")

    return account["account_id"].values, X, names


def loan_features(tables):
    loan, account = tables["loan"], tables["account"]
    dt = parse_flex_date(loan["date"])
    acct_district = dict(zip(account["account_id"], account["district_id"]))
    district_of_loan = loan["account_id"].map(acct_district)
    burst_key = pd.DataFrame({"district_id": district_of_loan, "date": loan["date"]})
    district_day_count = burst_key.groupby(["district_id", "date"])["date"].transform("size")

    X, names = pd.DataFrame(index=loan.index), []
    add_feature(X, names, "account_id_fk", loan["account_id"], "account_id")
    add_feature(X, names, "log_amount", np.log1p(loan["amount"].clip(lower=0)), "amount")
    add_feature(X, names, "duration", loan["duration"], "duration")
    add_feature(X, names, "payments", loan["payments"], "payments")
    add_feature(X, names, "day_of_month", dt.dt.day.fillna(0), "date")
    add_feature(X, names, "month", dt.dt.month.fillna(0), "date")
    # catches the district-wide loan burst: this many loans landed on the
    # same day, in the same district, as this one
    add_feature(X, names, "district_day_loan_count", district_day_count.fillna(1.0), "date")
    add_dummies(X, names, loan["status"], "status", "status")

    return loan["loan_id"].values, X, names


def order_features(tables):
    order = tables["order"]
    X, names = pd.DataFrame(index=order.index), []
    add_feature(X, names, "account_id_fk", order["account_id"], "account_id")
    add_feature(X, names, "log_amount", np.log1p(order["amount"].clip(lower=0)), "amount")
    add_dummies(X, names, order["bank_to"], "bank_to", "bank_to", dummy_na=True)
    add_dummies(X, names, order["k_symbol"], "k_symbol", "k_symbol", dummy_na=True)
    add_feature(X, names, "account_to", pd.to_numeric(order["account_to"], errors="coerce").fillna(-1.0), "account_to")

    return order["order_id"].values, X, names


def card_features(tables):
    card = tables["card"]
    issued_dt = parse_flex_date(card["issued"])
    n_per_disp = card.groupby("disp_id").size()

    X, names = pd.DataFrame(index=card.index), []
    add_feature(X, names, "disp_id_fk", card["disp_id"], "disp_id")
    add_dummies(X, names, card["type"], "type", "type")
    add_feature(X, names, "day_of_month", issued_dt.dt.day.fillna(0), "issued")
    add_feature(X, names, "month", issued_dt.dt.month.fillna(0), "issued")
    add_feature(X, names, "year", issued_dt.dt.year.fillna(0), "issued")
    # catches the per-disp card burst
    add_feature(X, names, "n_cards_per_disp", card["disp_id"].map(n_per_disp).astype(float), "disp_id")

    return card["card_id"].values, X, names


def client_features(tables):
    client = tables["client"]
    X, names = pd.DataFrame(index=client.index), []
    add_feature(X, names, "district_id_fk", client["district_id"], "district_id")
    add_feature(X, names, "birth_number", pd.to_numeric(client["birth_number"], errors="coerce"), "birth_number")

    return client["client_id"].values, X, names


def disp_features(tables):
    disp = tables["disp"]
    X, names = pd.DataFrame(index=disp.index), []
    add_feature(X, names, "client_id_fk", disp["client_id"], "client_id")
    add_feature(X, names, "account_id_fk", disp["account_id"], "account_id")
    add_dummies(X, names, disp["type"], "type", "type")

    return disp["disp_id"].values, X, names


def district_features(tables):
    district = tables["district"]
    X, names = pd.DataFrame(index=district.index), []
    for col in DISTRICT_NUMERIC_COLS:
        vals = pd.to_numeric(district[col], errors="coerce")  # some real copies use "?" for missing 1995 data
        add_feature(X, names, col, vals.fillna(vals.median()), col)
    add_dummies(X, names, district["region"], "region", "region")

    return district["district_id"].values, X, names


FEATURE_BUILDERS = {
    "trans": trans_features, "account": account_features, "loan": loan_features,
    "order": order_features, "card": card_features, "client": client_features,
    "disp": disp_features, "district": district_features,
}


# ----------------------------------------------------------------------------
def run_isolation_forest(X, contamination, seed, n_estimators):
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                             random_state=seed, n_jobs=-1)
    model.fit(X)
    score = -model.score_samples(X)      # sklearn returns -S(x,m); negate back to the paper's S(x,m)
    flag = model.predict(X) == -1        # sklearn's own contamination-based threshold
    return model, score, flag


def attribute_features(model, X, feature_names, top_n=3):
    """For each row, credit each feature by how much it shrank the
    population along that row's actual decision path in each tree --
    log(n_parent / n_child) per split, summed by originating feature.
    Isolation Forest splits on features uniformly at random, so raw split
    *frequency* carries no signal; this weights by how decisive each split
    actually was for isolating that specific row."""
    n_samples = X.shape[0]
    if n_samples == 0:
        return []
    credit = np.zeros((n_samples, X.shape[1]), dtype=np.float64)
    Xv = X.values

    for est, feat_idx in zip(model.estimators_, model.estimators_features_):
        tree = est.tree_
        node_counts = tree.n_node_samples
        node_indicator = est.decision_path(Xv[:, feat_idx])
        indptr, indices = node_indicator.indptr, node_indicator.indices
        for row in range(n_samples):
            path = indices[indptr[row]:indptr[row + 1]]
            for step in range(len(path) - 1):
                node, child = path[step], path[step + 1]
                f = tree.feature[node]
                if f < 0:
                    continue
                n_parent, n_child = node_counts[node], node_counts[child]
                if n_parent > 0 and n_child > 0:
                    credit[row, feat_idx[f]] += np.log(n_parent / n_child)

    # Several engineered features share one source column (log_amount and
    # amount_z are both `amount`), so their credit has to be SUMMED before
    # ranking. Ranking individual features first and de-duplicating the display
    # names afterwards lets a single mid-sized feature outrank a source column
    # that several features jointly dominate -- it reports the wrong cause.
    uniq = sorted(set(feature_names))
    col_of = np.array([uniq.index(n) for n in feature_names])
    by_col = np.zeros((n_samples, len(uniq)))
    np.add.at(by_col.T, col_of, credit.T)

    results = []
    for row in range(n_samples):
        ranked = np.argsort(-by_col[row])
        picked = [uniq[j] for j in ranked[:top_n] if by_col[row, j] > 0]
        results.append(picked or [uniq[ranked[0]]])
    return results


def structural_checks(tables):
    """Direct rule-based checks for the two fault types Isolation Forest has
    no real way to catch: a unique-key violation or a broken foreign key is
    a structural fact, not a statistical outlier, so check it exactly rather
    than hoping the model stumbles onto it."""
    records = []
    for table, pk in CHILD_PK.items():
        df = tables[table]
        dup_mask = df.duplicated(subset=[pk], keep=False)
        for rid in df.loc[dup_mask, pk].dropna().unique():
            records.append({"dataset": f"{table}.csv", "record_id": int(rid),
                             "score": 1.0, "features": ["duplicate_key"]})

    for child, fk_col, parent, parent_key in FK_SPECS:
        parent_df = tables[parent]
        if parent_key not in parent_df.columns:
            continue
        valid = set(parent_df[parent_key].dropna())
        child_df = tables[child]
        broken = child_df[~child_df[fk_col].isin(valid)]
        pk = CHILD_PK[child]
        for rid in broken[pk].dropna().unique():
            records.append({"dataset": f"{child}.csv", "record_id": int(rid),
                             "score": 1.0, "features": [fk_col]})
    return records


# ----------------------------------------------------------------------------
# Deterministic constraint checks
# ----------------------------------------------------------------------------
# Isolation Forest is a *global density* model over ~30 features that picks its
# split feature uniformly at random. A fault living in one feature therefore gets
# diluted by a factor of ~n_features, which is why it recovers only single-digit
# percentages of temporal and volume faults even when a one-line threshold on the
# right column recovers ~100% of them (measured: 6.9% vs 99.7%).
#
# Temporal violations and volume bursts are not distributional oddities, they are
# violated constraints -- exactly like a duplicate key or a dangling FK. So they
# belong here, checked exactly, and not left to the model to stumble onto.
#
# Balance corruption is the one genuinely statistical class (a swapped balance is
# not a violated constraint), so it stays a scored heuristic rather than a hard
# rule, and Isolation Forest keeps it as well.

# Faults are assumed rare, so the 0.1%/99.9% quantiles of a date column still sit
# inside the clean bulk. That derives the plausible window from the data rather
# than hardcoding Berka's 1993-1998 range.
DATE_QUANTILE = 0.001
# Per-account robust z (median/MAD) above which a balance is called corrupt.
# 10 is the knee of the recall/precision curve on injected data (49% / 56%);
# lower floods the output, higher trades away most of the recall.
BALANCE_ROBUST_Z = 10.0
# A burst is "more same-day events on one parent than the clean data ever shows".
# Derived per table from its own group-size distribution rather than fixed.
BURST_QUANTILE = 0.9999
# Pre-open transactions on one account before the ACCOUNT's own date is blamed
# rather than the transactions'. See the note in constraint_checks().
MIN_PREOPEN_TXNS = 2


def _robust_z(values, group):
    """Per-group |median-centred z| using MAD, which a corrupted row cannot drag
    the way mean/std lets it mask itself."""
    med = values.groupby(group).transform("median")
    mad = (values - med).abs().groupby(group).transform("median")
    return ((values - med) / mad.replace(0, np.nan)).abs().fillna(0.0)


def _burst_threshold(group_sizes):
    """Largest same-day group the clean data plausibly produces.

    Must be computed on the GROUP SIZES, not on the per-row transform of them:
    a burst of k rows contributes k copies of the value k, so the transform
    inflates its own tail quadratically and the quantile lands exactly ON the
    burst value -- making `> threshold` match nothing. (Measured: on the per-row
    series quantile(.9999) == 31 == the injected burst size, recall 8%.)
    """
    return max(float(group_sizes.quantile(BURST_QUANTILE)), 1.0)


def constraint_checks(tables):
    """Exact checks for the fault classes that are constraint violations rather
    than density outliers. Returns records in the same shape as
    structural_checks()."""
    records = []

    def emit(table, ids, feature):
        for rid in pd.Series(ids).dropna().unique():
            records.append({"dataset": f"{table}.csv", "record_id": int(rid),
                            "score": 1.0, "features": [feature]})

    account = tables["account"]
    acct_open = dict(zip(account["account_id"], parse_flex_date(account["date"])))

    # --- temporal: a child event before its parent account existed, or a date
    #     outside the range the bulk of the table occupies ---
    bad_dates = {}
    for table, date_col, parent_of in (
        ("trans", "date", lambda df: df["account_id"]),
        ("loan", "date", lambda df: df["account_id"]),
        ("card", "issued", lambda df: df["disp_id"].map(
            dict(zip(tables["disp"]["disp_id"], tables["disp"]["account_id"])))),
    ):
        df = tables[table]
        dt = parse_flex_date(df[date_col])
        open_dt = pd.to_datetime(parent_of(df).map(acct_open))
        before_parent = (dt < open_dt).fillna(False)
        # Quantile the YEAR, not the timestamp. A quantile of a continuous date
        # always has data beyond it, so a raw-timestamp fence flags a fixed
        # 2*DATE_QUANTILE of *any* input, clean or not (measured: 1,054 false
        # positives on untouched Berka). Snapping to whole years makes the fence
        # land on a year boundary, so clean data produces none.
        year = dt.dt.year
        lo, hi = year.quantile(DATE_QUANTILE), year.quantile(1 - DATE_QUANTILE)
        out_of_range = ((year < lo) | (year > hi)).fillna(False)
        if table == "trans":
            bad_dates["trans_before_parent"] = before_parent
        emit(table, df.loc[before_parent | out_of_range, CHILD_PK[table]], date_col)

    # "transaction predates its account" and "account opened after its own first
    # transaction" are the SAME violation seen from opposite sides, and nothing in
    # the data says which side is corrupt. Corrupting one transaction produces
    # exactly one violation, so a *second* one on the same account is evidence
    # that the account's open date moved instead. Requiring two catches all five
    # injected account faults for 48 extra account flags; flagging on one flags
    # 551 of 4,504 accounts, almost all of them collateral from trans-side faults.
    #
    # The transactions stay flagged either way. Withholding them would trade
    # ~90 real trans-side faults for 5 account-side ones, and when two rows
    # predate an account's open date both records genuinely warrant review.
    txn = tables["trans"]
    n_before = bad_dates["trans_before_parent"].groupby(txn["account_id"]).transform("sum")
    emit("account", txn.loc[n_before >= MIN_PREOPEN_TXNS, "account_id"], "date")

    # --- volume: more same-day events on one parent than the clean data shows ---
    def emit_bursts(table, keys, pk, feature):
        g = tables[table].groupby(keys)
        thr = _burst_threshold(g.size())
        per_row = g[pk].transform("size")
        emit(table, tables[table].loc[per_row > thr, pk], feature)

    trans = tables["trans"]
    emit_bursts("trans", ["account_id", "date"], "trans_id", "account_day_txn_count")
    emit_bursts("card", ["disp_id"], "card_id", "n_cards_per_disp")

    # loans burst across a district, so the group key lives on account, not loan
    loan = tables["loan"]
    dist_of = dict(zip(account["account_id"], account["district_id"]))
    loan_key = pd.DataFrame({"d": loan["account_id"].map(dist_of), "date": loan["date"]})
    loan_grp = loan_key.groupby(["d", "date"])["date"]
    emit("loan", loan.loc[loan_grp.transform("size") > _burst_threshold(loan_grp.size()),
                          "loan_id"], "district_day_loan_count")

    # --- balance: scored heuristic, not a hard rule (see module note above) ---
    z = _robust_z(trans["balance"], trans["account_id"])
    emit("trans", trans.loc[z > BALANCE_ROBUST_Z, "trans_id"], "balance_robust_z")

    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="./data/raw")
    p.add_argument("--out", default="./artifacts_faulty")
    p.add_argument("--contamination", type=float, default=0.005,
                   help="expected anomaly fraction per table, passed straight to IsolationForest")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--top-n-features", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tables", default=None,
                   help="comma-separated subset of tables to run the ML pass on, e.g. "
                        "'trans,loan,card'. Structural checks always run on all tables. Default: all.")
    args = p.parse_args()

    selected = TABLES if args.tables is None else [t.strip() for t in args.tables.split(",")]
    unknown = set(selected) - set(TABLES)
    if unknown:
        p.error(f"unknown table(s): {unknown}. choose from {TABLES}")

    print("=== loading ===")
    tables = load_raw(args.raw)
    for name, df in tables.items():
        print(f"  {name}: {len(df):,} rows")

    print("\n=== rule-based structural checks ===")
    all_records = structural_checks(tables)
    print(f"  duplicate keys + broken foreign keys: {len(all_records)} flagged")

    print("\n=== deterministic constraint checks ===")
    constraint_records = constraint_checks(tables)
    all_records += constraint_records
    print(f"  temporal / volume / balance constraints: {len(constraint_records)} flagged")

    print("\n=== isolation forest per table ===")
    for table in selected:
        builder = FEATURE_BUILDERS[table]
        record_ids, X, display_names = builder(tables)
        X = X.fillna(0.0).astype(float)
        if len(X) < 10:
            print(f"  {table}: only {len(X)} rows, skipping ML pass")
            continue

        model, score, flag = run_isolation_forest(X, args.contamination, args.seed, args.n_estimators)
        flagged_idx = np.where(flag)[0]
        print(f"  {table}: {len(X):,} rows scored, {len(flagged_idx)} flagged "
              f"(score range {score.min():.4f}-{score.max():.4f})")

        if len(flagged_idx) == 0:
            continue
        features_per_row = attribute_features(model, X.iloc[flagged_idx], display_names, args.top_n_features)
        for i, row_i in enumerate(flagged_idx):
            all_records.append({"dataset": f"{table}.csv", "record_id": int(record_ids[row_i]),
                                 "score": float(score[row_i]), "features": features_per_row[i]})

    # merge duplicate (dataset, record_id) pairs -- a row can be flagged by
    # both the rule-based pass and the ML pass; keep the max score and the
    # union of features rather than reporting it twice
    merged = {}
    for r in all_records:
        key = (r["dataset"], r["record_id"])
        if key not in merged:
            merged[key] = {"dataset": r["dataset"], "record_id": r["record_id"],
                            "score": r["score"], "features": list(r["features"])}
        else:
            merged[key]["score"] = max(merged[key]["score"], r["score"])
            for f in r["features"]:
                if f not in merged[key]["features"]:
                    merged[key]["features"].append(f)

    final = sorted(merged.values(), key=lambda r: -r["score"])
    anomalies = [{"anomaly_id": f"ANOM-{i + 1:05d}", "dataset": r["dataset"],
                  "record_id": r["record_id"], "score": round(r["score"], 4),
                  "features": r["features"]}
                 for i, r in enumerate(final)]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "anomalies.json"), "w") as fh:
        json.dump(anomalies, fh, indent=2)

    print(f"\n=== output ===\n  {os.path.join(args.out, 'anomalies.json')}: {len(anomalies)} anomalies")
    if anomalies:
        print(f"  top score: {anomalies[0]['score']} ({anomalies[0]['dataset']}, "
              f"record_id {anomalies[0]['record_id']})")


if __name__ == "__main__":
    main()
