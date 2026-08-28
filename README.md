# Anomaly detection on the Berka / PKDD'99 banking dataset

Unsupervised anomaly detection with Isolation Forest, at two granularities:
individual transactions and whole accounts. Standalone — pandas, numpy and
scikit-learn only. No database, no graph engine.

## Folder layout

Everything is relative to the repo root. Only `data/raw/` needs to be put there
by hand; `artifacts/` is created for you.

```
capstone/
├── detect_anomalies.py            <- the script
├── requirements-anomaly.txt
│
├── data/
│   └── raw/                       <- YOU PUT THE 8 CSVs HERE (not in git)
│       ├── account.csv
│       ├── card.csv
│       ├── client.csv
│       ├── disp.csv
│       ├── district.csv
│       ├── loan.csv
│       ├── order.csv
│       └── trans.csv
│
└── artifacts/                     <- created by the script; outputs land here
    ├── anomalies.json
    ├── anomaly_scores_trans.csv
    ├── anomaly_scores_account.csv
    └── anomaly_by_district.csv
```

### Input: `data/raw/`

The eight raw CSVs, **semicolon-delimited, unmodified** as distributed. Do not
rename them — the script loads them by exact filename. `district.csv` ships with
opaque `A1..A16` headers; the script renames those itself, so leave it alone.

| File | Rows | Notes |
|---|---|---|
| `account.csv` | 4,500 | |
| `card.csv` | 892 | |
| `client.csv` | 5,369 | |
| `disp.csv` | 5,369 | links clients to accounts |
| `district.csv` | 77 | `A1..A16` headers, renamed on load |
| `loan.csv` | 682 | `status` is held out as validation, not a feature |
| `order.csv` | 6,471 | |
| `trans.csv` | 1,056,320 | 66 MB — the big one |

`data/raw/` is **gitignored** (67 MB total). A fresh clone will not have it —
get the CSVs from the PKDD'99 dataset or from whoever shared the repo, and drop
them in before running.

Pass `--raw` if your CSVs live somewhere else:

```bash
python detect_anomalies.py --raw /path/to/csvs
```

### Output: `artifacts/`

Created automatically. All four files are small enough to commit and diff.

| File | Size | Records | Contents |
|---|---|---|---|
| `anomalies.json` | 1.0 MB | 5,372 | one object per anomaly, with the features that isolated it |
| `anomaly_scores_trans.csv` | 539 KB | 5,282 | flagged transactions, most anomalous first |
| `anomaly_scores_account.csv` | 223 KB | 4,500 | every account, scored |
| `anomaly_by_district.csv` | 6 KB | 77 | per-district anomaly rate and lift |

The transaction file holds **flagged rows only**. The full 1.06M-row score table
is ~101 MB and is deliberately not written by default; use `--all-scores` if you
need it for threshold tuning, and don't commit the result.

### `anomalies.json`

```json
{
  "anomaly_id": "ANOM-00001",
  "dataset": "trans.csv",
  "record_id": 753736,
  "score": 1.0,
  "features": ["date", "balance", "amount"]
}
```

`score` is min-maxed **within its own dataset**, so a transaction's 1.0 and an
account's 1.0 are different quantities. Records are grouped by dataset for that
reason — transactions first (`ANOM-00001`..`ANOM-05282`), then accounts — each
block sorted most-anomalous-first. Do not read the ordering as a single global
ranking across both.

`features` names the source columns the forest actually used to isolate that
record (see *Feature attribution* below). For `trans.csv` records these are
columns of trans.csv; for `account.csv` records they are aggregate feature names
(`n_txn`, `amount_std`, ...), which have no single source column.

## Setup and run

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements-anomaly.txt
```

```bash
python detect_anomalies.py --raw ./data/raw --out ./artifacts
```

Runs in about 30 seconds. Prints the validation report and district rollup to
the terminal, then writes the three CSVs.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--raw` | `./data/raw` | where the 8 input CSVs live |
| `--out` | `./artifacts` | where outputs are written |
| `--contamination` | `0.005` | expected anomaly fraction — a budget you choose, not a measurement |
| `--seed` | `42` | random seed |
| `--all-scores` | off | write all 1.06M transaction scores (~101 MB) |

The account model uses `4 x --contamination`, because accounts are 235x rarer
than transactions and the same fraction would flag only ~22 of them.

## Results, and how far to trust them

Berka has **no fraud labels**. `--contamination 0.005` is why there are 5,282
flags: you asked for 0.5%, you got 0.5%. So the models are validated by
triangulation, not by accuracy.

**The account model holds up.** Loan status is excluded from the feature matrix
entirely, then checked afterwards. Flagged accounts default at 26.67% against an
11.14% base rate — a **2.39x lift** on a signal the model never saw.

**The transaction model is weaker, and should be read with that in mind.** It
catches only 21 of 2,999 negative-balance rows (0.7%), because `log_amount` and
`balance` dominate the splits — every top-ranked anomaly is a large
`PREVOD Z UCTU` credit. It surfaces 5,261 rows no balance rule catches, which is
the point of running it, but it is **not** a superset of a `balance < 0` rule and
shouldn't be presented as one. To weight solvency over raw magnitude, drop the
raw magnitude columns and lean on `amount_z`, `balance_ratio` and
`balance_delta`.

**District results are volume-normalised.** Ranking districts by raw anomaly
count just ranks them by size — Prague comes first with 631 anomalies but sits
*below* baseline at 0.476%. By rate, the real outliers are Praha-zapad (2.24x),
Plzen-jih (1.96x) and Chomutov (1.69x).

## Features

Per-account-relative features are what make this worth doing over a threshold: a
50K withdrawal is unremarkable globally and glaring on an account whose every
other movement is under 2K.

**Transaction (26 features)** — `log_amount`, `balance`, `balance_delta`,
`amount_z` (z-score within the account's own history), `balance_ratio`,
`days_since_prev`, `day_of_month`, `month`, `is_interbank`, plus one-hot `type`
(3), `operation` (6) and `k_symbol` (9).

**Account (16 features)** — transaction count, mean/std/max amount,
mean/min/max/std balance, fraction of negative balances, fraction of credits,
lifespan in days, transactions per day, and product holdings (orders, loans,
cards, clients). Loan **status** is excluded on purpose.

No feature scaling: Isolation Forest splits one feature at a time and is
invariant to per-feature monotone rescaling.

### Feature attribution

Isolation Forest has no `feature_importances_` and no per-instance attribution —
`score_samples` returns one scalar and nothing about why. The `features` field in
`anomalies.json` is reconstructed from the isolation paths.

The obvious approach does not work, and it is worth knowing why before
"improving" it: crediting whichever feature each split used (weighted by depth)
returns **the same three features for nearly every record**. Isolation Forest
picks its split feature *uniformly at random*, independently of the data, so
which features appear on a path is near-identical for anomalous and normal rows.

What carries the signal is not which feature split, but how much isolation that
split achieved. A split dropping a row from 256 candidate samples to 3 did the
isolating; one going 256 → 128 did nothing. Each split is therefore credited
`log(n_parent / n_child)` along the path the row actually took, summed over all
200 trees and normalised to a share. Engineered features are then collapsed onto
their source column (`log_amount` and `amount_z` both count toward `amount`).

Sanity checks that this tracks the data rather than the tree structure:

| Check | Result |
|---|---|
| transactions with `amount > 60k` citing `amount` | 64.7% |
| transactions with `amount < 10k` citing `amount` | 0.8% |
| pension (`DUCHOD`) rows citing `k_symbol` | 100% |

This is a construction of this project, not a standard library output. For a
citable alternative, SHAP's `TreeExplainer` supports IsolationForest.

## Other scripts in this repo

`build_offline_lineage.py`, `export_to_gephi.py` and `render_offline_graph.py`
are a separate lineage-graph track and need `kuzu`, `networkx` and `pyvis`
— none of which `detect_anomalies.py` requires. `requirements-anomaly.txt`
covers the anomaly script only.
