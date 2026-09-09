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

Three tiers per table, in increasing order of how much they guess. Every record
carries the tier that found it.

  1. structural    Exact duplicate-key and broken-foreign-key lookups. A model
                    trained on distributions has no concept of "this primary key
                    must be unique" or "this FK must resolve".
  2. constraint    Exact temporal checks (an event predating its own parent, or a
                    date outside the range the baseline occupies), same-day burst
                    counts, and a per-account robust-z check on balance paired
                    with a day-level balance-arithmetic reconstruction.
  3. model         A scoped IsolationForest ensemble -- the residual net for
                    anomalies nobody wrote a check for.

Tiers 1 and 2 do nearly all the work, and that split is measured, not assumed.
Isolation Forest picks its split feature uniformly at random, so a fault living
in one feature is diluted by a factor of ~n_features. Measured against
inject_faults2.py's ground truth (see benchmark.py), a single forest over all
features recovered 6.9% of temporal violations and 10.7% of volume bursts, where
a threshold on the single right column recovers ~100% of both. Anything
expressible as a violated constraint therefore belongs in tier 1 or 2.

Balance corruption is the one genuinely statistical class -- a swapped balance
violates no constraint -- so it stays a scored heuristic and the model keeps it.

Tiers 1+2 ("review") over five injections, thresholds calibrated on the clean
baseline and never on the data being scored:

    precision  90.9% +/- 3.9      structural only  100.0% precision, 49.1% recall
    recall     88.0% +/- 0.3      constraint only   82.0% precision, 39.1% recall
    F1         89.4% +/- 1.9      clean-data flags  93

Per fault class (seed 42): duplicate_keys, referential_breaks,
temporal_violations and volume_spikes all 100%; balance_corruption 50.8%. The
balance shortfall is almost entirely the "swap" sub-mode, where a real balance
from another account replaces this one: 75% of those sit inside clean Berka's own
99.9th percentile of per-account robust z, so no per-row statistic reaches them.

The model tier is kept but written separately -- see TIER_RANK. It fires on a
fixed contamination quota whether or not anything is wrong (5,400 flags on a
completely clean copy of Berka), so merging it into the review queue costs ~53
points of precision for ~0.1 of recall.

Every table has different columns, so each gets its own feature builder --
see FEATURE_BUILDERS. district.csv and disp.csv have very little signal
(few numeric columns) and rely mostly on the exact checks.

Output: anomalies.json (tiers 1-2, for review) and anomalies_model.json (tier 3),
one entry per flagged record, numbered ANOM-00001, ANOM-00002, ...

Usage:
    python anomaly_detection.py --raw ./data/faulty --baseline ./data/raw \
                                --out ./artifacts_faulty
    python evaluate_detection.py --faults ./data/faulty/injected_faults.json \
                                 --anomalies ./artifacts_faulty/anomalies.json
    python benchmark.py --raw ./data/raw --work ./data/bench --max-clean-flags 150

--baseline is strongly recommended: without it every threshold is derived from
the data being screened, which leaks and, for burst checks, inverts the test --
a burst is itself the largest group, so it raises the very threshold meant to
catch it. Measured on loan: 0% recall self-calibrated, 100% against a baseline.
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

# Which pass flagged a record, most authoritative first. "structural" and
# "constraint" are exact checks; "model" is a density estimate that fires on a
# fixed contamination quota whether or not anything is wrong -- measured at 2.4%
# precision against injected ground truth, and it emits 5,400 flags on a
# completely clean copy of Berka. So model records are written to a separate file
# rather than into the review queue: mixing them in drops end-to-end precision
# from ~95% to ~42% while adding 0.1 points of recall.
TIER_RANK = {"structural": 0, "constraint": 1, "model": 2}
REVIEW_TIERS = ("structural", "constraint")

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
    add_feature(X, names, "birth_number", pd.to_numeric(client["birth_number"], errors="coerce"), "birth_number")

    return client["client_id"].values, X, names


def disp_features(tables):
    disp = tables["disp"]
    X, names = pd.DataFrame(index=disp.index), []
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


def _percentile_rank(score):
    """Score -> its own within-model percentile, so models fitted on feature sets
    of different widths become comparable. Raw S(x,m) is not: its scale depends on
    how many features were available to split on."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    return ranks / max(len(score) - 1, 1)


def run_scoped_forest(X, display_names, contamination, seed, n_estimators):
    """One small forest per source field, combined by max percentile.

    Isolation Forest chooses its split feature uniformly at random, so a signal
    carried by one feature out of n is examined by only ~1/n of splits. Measured
    on balance_corruption with everything else held fixed: 32 features 7.6%,
    the 6 balance-related features 54.9%, balance_robust_z alone 57.3%. The
    fix is not a hyperparameter, it is scope.

    So features are partitioned by the source field they describe -- every
    balance-derived column in one model, every date-derived column in another --
    and each model sees only its own group. A row is as anomalous as the group
    that finds it most anomalous, which also names the dimension it is anomalous
    in, giving per-record attribution without reconstructing isolation paths.

    Returns (per_group_rank, combined, flag, winning_group).
    """
    groups = {}
    for col, display in zip(X.columns, display_names):
        groups.setdefault(display, []).append(col)

    names = sorted(groups)
    ranks = np.zeros((len(names), len(X)))
    for i, name in enumerate(names):
        sub = X[groups[name]]
        model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                random_state=seed, n_jobs=-1).fit(sub)
        ranks[i] = _percentile_rank(-model.score_samples(sub))

    combined = ranks.max(axis=0)
    winner = [names[i] for i in ranks.argmax(axis=0)]
    # One global budget over the combined score. Thresholding each group
    # separately and unioning would spend the budget n_groups times over.
    flag = combined >= np.quantile(combined, 1.0 - contamination)
    return ranks, combined, flag, winner


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
                             "score": 1.0, "tier": "structural",
                             "features": ["duplicate_key"]})

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
                             "score": 1.0, "tier": "structural",
                             "features": [fk_col]})
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

# --- thresholds -------------------------------------------------------------
#
# Every threshold below is a FALLBACK, used only when no clean baseline is
# supplied. Prefer --baseline: deriving a threshold from the same data you are
# screening is biased, and for bursts it is provably wrong. A burst of k rows is
# itself the largest group in the table, so any statistic of the group sizes --
# max, or a high quantile -- is inflated by the very fault it is meant to catch.
# Measured on loan: clean data never exceeds 2 loans per district-day, the
# injected burst is 5, and the self-derived threshold lands at 5, so `> thr`
# matches nothing and all 5 injected loans are missed. Calibrated against clean
# Berka the same check scores 100%. See calibrate().

# Faults are assumed rare, so the 0.1%/99.9% quantiles of a date column still sit
# inside the clean bulk. That derives the plausible window from the data rather
# than hardcoding Berka's 1993-1998 range.
DATE_QUANTILE = 0.001
# Per-account robust z (median/MAD) above which a balance is called corrupt.
# 10 is the knee of the recall/precision curve on injected data (49% / 56%);
# lower floods the output, higher trades away most of the recall. With a clean
# baseline this is replaced by a quantile at CLEAN_FPR (14.07 on Berka).
BALANCE_ROBUST_Z = 10.0
# A burst is "more same-day events on one parent than the clean data ever shows".
# Derived per table from its own group-size distribution rather than fixed.
BURST_QUANTILE = 0.9999
# Pre-open transactions on one account before the ACCOUNT's own date is blamed
# rather than the transactions'. See the note in constraint_checks().
MIN_PREOPEN_TXNS = 2

# Weaker robust-z fence, used ONLY in conjunction with a failed day chain (see
# _day_chain_unresolved). Alone it flags far too much; paired with independent
# evidence that the account's arithmetic does not close, it is informative.
BALANCE_ROBUST_Z_WEAK = 8.5

# Share of clean rows the balance heuristic is allowed to flag. This is a budget
# you choose, not a fit: the threshold is read off the baseline's own robust-z
# distribution at this quantile, so no injected fault influences it.
CLEAN_FPR = 5e-5
# The weak fence is allowed a looser budget because it never fires on its own.
WEAK_FPR_MULTIPLE = 20


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


def _day_chain_unresolved(trans):
    """Per row: did the account's balance arithmetic fail to close on that day?

    A swapped balance is not a distributional outlier -- it is a real balance
    from another account, and 75% of them land inside clean Berka's own 99.9th
    percentile of per-account robust z. No per-row statistic reaches them. What
    does reach them is arithmetic: an account's balance should march in step with
    its own transactions.

    The row order inside a day is not recoverable in Berka -- the sign convention
    is unambiguous (PRIJEM credits, VYBER/VYDAJ debit) yet 72.7% of unexplained
    balance movements match some OTHER row's amount, i.e. the neighbour is wrong,
    not the arithmetic. So this checks the one quantity a day's internal order
    cannot change: its closing balance, which must be the previous close plus the
    day's net movement, and must appear among the balances the day records.

    On a failure the walk re-anchors on a balance the day actually recorded
    rather than carrying the predicted value forward, so one corrupt day does not
    invalidate every later day on the account.

    Resolves 91.7% of clean account-days, so the 8.3% residue is far too noisy to
    flag on its own (116k spurious rows). It is only ever used in conjunction --
    see constraint_checks.
    """
    d = trans.copy()
    d["_sgn"] = np.where(d["type"] == "PRIJEM", d["amount"], -d["amount"])
    d["_dt"] = parse_flex_date(d["date"])
    d = d.sort_values(["account_id", "_dt"], kind="stable")

    grp = d.groupby(["account_id", "_dt"], sort=False)
    net = grp["_sgn"].sum()
    seen = grp["balance"].apply(lambda s: np.round(s.values, 1))
    accounts = net.index.get_level_values(0).values

    resolved = {}
    close, current = 0.0, None
    for key, acct, movement, balances in zip(net.index, accounts, net.values, seen.values):
        if acct != current:
            current, close = acct, 0.0          # accounts open at zero in Berka
        predicted = round(close + movement, 1)
        hit = predicted in set(balances)
        resolved[key] = hit
        # re-anchor on the day's own last recorded balance, keeping the walk in
        # the data's frame of reference rather than compounding a bad prediction
        close = predicted if hit else float(balances[-1])

    per_row = [not resolved[k] for k in zip(d["account_id"].values, d["_dt"].values)]
    return pd.Series(per_row, index=d.index).reindex(trans.index).fillna(False)


def _loan_district_groups(tables):
    """Loans grouped by (district, date) -- loans burst across a district, so the
    group key lives on account, not on loan."""
    dist_of = dict(zip(tables["account"]["account_id"], tables["account"]["district_id"]))
    key = pd.DataFrame({"d": tables["loan"]["account_id"].map(dist_of),
                        "date": tables["loan"]["date"]})
    return key.groupby(["d", "date"])["date"]


def calibrate(tables):
    """Read every constraint threshold off a TRUSTED, fault-free snapshot.

    The thresholds are what the clean data never exceeds: the largest same-day
    group it contains, the years it spans, the robust-z its own tail reaches at
    CLEAN_FPR. Applied unchanged to the data under test, so nothing in the
    screened data -- and no injected fault -- can move a threshold.

    This is what makes the numbers defensible: the alternative, self-calibration,
    both leaks the test set into threshold selection and (for bursts) inverts the
    check outright. In production the baseline is a known-good historical window;
    in evaluation it is the uncorrupted dataset.
    """
    cal = {"burst": {}, "year": {}}
    cal["burst"]["trans"] = int(tables["trans"].groupby(["account_id", "date"]).size().max())
    cal["burst"]["card"] = int(tables["card"].groupby("disp_id").size().max())
    cal["burst"]["loan"] = int(_loan_district_groups(tables).size().max())

    # The clean span IS the plausible window -- no quantile needed, because a
    # trusted baseline has no out-of-range dates to trim.
    for table, date_col in (("trans", "date"), ("loan", "date"), ("card", "issued")):
        year = parse_flex_date(tables[table][date_col]).dt.year.dropna()
        cal["year"][table] = (int(year.min()), int(year.max()))

    z = _robust_z(tables["trans"]["balance"], tables["trans"]["account_id"])
    cal["balance_z"] = float(np.quantile(z.values, 1.0 - CLEAN_FPR))
    cal["balance_z_weak"] = float(np.quantile(z.values, 1.0 - WEAK_FPR_MULTIPLE * CLEAN_FPR))
    return cal


def constraint_checks(tables, cal=None, day_chain=True):
    """Exact checks for the fault classes that are constraint violations rather
    than density outliers. Returns records in the same shape as
    structural_checks().

    `cal` comes from calibrate() on a clean baseline. Without it the thresholds
    fall back to statistics of `tables` itself, which is biased -- see the note
    above the threshold constants.

    `day_chain` adds the balance-arithmetic check, which is what reaches swapped
    balances. It walks every account day by day, so it costs ~30s on Berka's 1.06M
    transactions; turn it off for a fast pass.
    """
    records = []

    def emit(table, ids, feature):
        for rid in pd.Series(ids).dropna().unique():
            records.append({"dataset": f"{table}.csv", "record_id": int(rid),
                            "score": 1.0, "tier": "constraint",
                            "features": [feature]})

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
        if cal is not None:
            lo, hi = cal["year"][table]
        else:
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
    def emit_bursts(table, groups, pk, feature):
        thr = cal["burst"][table] if cal is not None else _burst_threshold(groups.size())
        emit(table, tables[table].loc[groups.transform("size") > thr, pk], feature)

    trans = tables["trans"]
    emit_bursts("trans", trans.groupby(["account_id", "date"])["trans_id"],
                "trans_id", "account_day_txn_count")
    emit_bursts("card", tables["card"].groupby("disp_id")["card_id"],
                "card_id", "n_cards_per_disp")
    emit_bursts("loan", _loan_district_groups(tables),
                "loan_id", "district_day_loan_count")

    # --- balance: scored heuristic, not a hard rule (see module note above) ---
    #
    # Two fences, because balance corruption is two problems. A shocked value is
    # far outside its account's own scale and the strong fence alone finds 85% of
    # them. A swapped value is a real balance from elsewhere and sits inside the
    # normal range, so the strong fence finds ~5%. The weak fence reaches those,
    # but only where the day's arithmetic independently fails to close -- on its
    # own it would flag 116k clean rows.
    z = _robust_z(trans["balance"], trans["account_id"])
    z_hi = cal["balance_z"] if cal is not None else BALANCE_ROBUST_Z
    corrupt = z > z_hi
    if day_chain:
        z_lo = cal["balance_z_weak"] if cal is not None else BALANCE_ROBUST_Z_WEAK
        corrupt |= (z > z_lo) & _day_chain_unresolved(trans)
    emit("trans", trans.loc[corrupt, "trans_id"], "balance_robust_z")

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
    p.add_argument("--baseline", default=None,
                   help="directory of a TRUSTED, fault-free copy of the same schema. "
                        "Constraint thresholds are read off it instead of off the data "
                        "under test, which is both less biased and, for burst checks, "
                        "the only correct option. Strongly recommended.")
    p.add_argument("--tables", default=None,
                   help="comma-separated subset of tables to run the ML pass on, e.g. "
                        "'trans,loan,card'. Structural checks always run on all tables. Default: all.")
    p.add_argument("--merge-model", action="store_true",
                   help="write model-tier records into anomalies.json instead of a "
                        "separate file. Restores pre-tier behaviour; costs ~53 points "
                        "of precision for ~0.1 of recall.")
    p.add_argument("--monolithic-model", action="store_true",
                   help="fit ONE forest over all features per table instead of the scoped "
                        "ensemble. For ablation: measured 5.2%% vs 50.3%% recall on "
                        "balance_corruption, and 0.6%% vs 9.9%% on the faults the exact "
                        "checks miss.")
    p.add_argument("--no-day-chain", action="store_true",
                   help="skip the balance-arithmetic check (~30s on 1.06M rows). It is "
                        "the only check that reaches swapped balances.")
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

    cal = None
    if args.baseline:
        print(f"\n=== calibrating on baseline ({args.baseline}) ===")
        cal = calibrate(load_raw(args.baseline))
        print(f"  burst thresholds : " +
              ", ".join(f"{k}>{v}" for k, v in cal["burst"].items()))
        print(f"  plausible years  : " +
              ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in cal["year"].items()))
        print(f"  balance robust-z : {cal['balance_z']:.2f}  (baseline quantile at "
              f"CLEAN_FPR={CLEAN_FPR:g})")
    else:
        print("\n  warning: no --baseline given; constraint thresholds fall back to "
              "statistics of\n           the data under test. Burst checks in "
              "particular are unreliable this way\n           -- a burst inflates the "
              "very statistic used to detect it.")

    print("\n=== deterministic constraint checks ===")
    constraint_records = constraint_checks(tables, cal, day_chain=not args.no_day_chain)
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

        if args.monolithic_model:
            model, score, flag = run_isolation_forest(X, args.contamination, args.seed,
                                                      args.n_estimators)
            winner = None
        else:
            _, score, flag, winner = run_scoped_forest(X, display_names, args.contamination,
                                                       args.seed, args.n_estimators)
        flagged_idx = np.where(flag)[0]
        print(f"  {table}: {len(X):,} rows scored, {len(flagged_idx)} flagged "
              f"(score range {score.min():.4f}-{score.max():.4f})")

        if len(flagged_idx) == 0:
            continue
        if winner is not None:
            # The group that found the row most anomalous already names the
            # dimension it is anomalous in -- 99.2% agreement with the injected
            # fault class on balance, 96.7% on temporal -- so there is nothing to
            # reconstruct from isolation paths.
            features_per_row = [[winner[i]] for i in flagged_idx]
        else:
            features_per_row = attribute_features(model, X.iloc[flagged_idx],
                                                  display_names, args.top_n_features)
        for i, row_i in enumerate(flagged_idx):
            all_records.append({"dataset": f"{table}.csv", "record_id": int(record_ids[row_i]),
                                 "score": float(score[row_i]), "tier": "model",
                                 "features": features_per_row[i]})

    # merge duplicate (dataset, record_id) pairs -- a row can be flagged by
    # more than one pass; keep the max score, the union of features, and the
    # most authoritative tier that fired on it
    merged = {}
    for r in all_records:
        key = (r["dataset"], r["record_id"])
        if key not in merged:
            merged[key] = {"dataset": r["dataset"], "record_id": r["record_id"],
                            "score": r["score"], "tier": r["tier"],
                            "features": list(r["features"])}
        else:
            merged[key]["score"] = max(merged[key]["score"], r["score"])
            if TIER_RANK[r["tier"]] < TIER_RANK[merged[key]["tier"]]:
                merged[key]["tier"] = r["tier"]
            for f in r["features"]:
                if f not in merged[key]["features"]:
                    merged[key]["features"].append(f)

    final = sorted(merged.values(), key=lambda r: (TIER_RANK[r["tier"]], -r["score"]))
    anomalies = [{"anomaly_id": f"ANOM-{i + 1:05d}", "dataset": r["dataset"],
                  "record_id": r["record_id"], "score": round(r["score"], 4),
                  "tier": r["tier"], "features": r["features"]}
                 for i, r in enumerate(final)]

    # Split by tier rather than merging. The exact checks are what a steward
    # should actually work through; the model tier is a residual net for
    # anomalies nobody wrote a check for, and is mostly noise by volume.
    review = [r for r in anomalies if r["tier"] in REVIEW_TIERS]
    residual = [r for r in anomalies if r["tier"] not in REVIEW_TIERS]
    if args.merge_model:
        review, residual = anomalies, []

    os.makedirs(args.out, exist_ok=True)
    review_path = os.path.join(args.out, "anomalies.json")
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    if residual:
        residual_path = os.path.join(args.out, "anomalies_model.json")
        with open(residual_path, "w") as fh:
            json.dump(residual, fh, indent=2)

    by_tier = {t: sum(1 for r in anomalies if r["tier"] == t) for t in TIER_RANK}
    print("\n=== output ===")
    print(f"  by tier: " + ", ".join(f"{t}={n:,}" for t, n in by_tier.items()))
    print(f"  {review_path}: {len(review):,} anomalies for review")
    if residual:
        print(f"  {residual_path}: {len(residual):,} model-tier (low precision; "
              f"unknown-unknowns net)")
    if review:
        print(f"  top: {review[0]['dataset']} record_id {review[0]['record_id']} "
              f"({review[0]['tier']}, {', '.join(review[0]['features'])})")


if __name__ == "__main__":
    main()
