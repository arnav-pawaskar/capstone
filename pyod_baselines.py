"""
PyOD baselines on the trans table, scored against the injected-fault log.

Install (pin and report versions in the paper):
    pip install pyod==<version> scikit-learn pandas numpy
    # AutoEncoder needs a deep-learning backend (torch for pyod 2.x)

Usage:
    python pyod_baselines.py --manifest manifest.json --out pyod_results.csv \
        --variants full core6 --budgets 0.005 4139

Design notes (state these in the paper):
  * Each model is fit on a random subsample (--fit-sample) of the data under test,
    then scores ALL rows. Models rank rows; we flag the top-k by decision_function.
  * Budgets: a value < 1 is a fraction of rows (0.005 = the paper's contamination);
    a value >= 1 is an absolute flag count (e.g. 4139 = the review-queue size at
    seed 42), so precision/recall are compared at an equal reviewer cost.
  * Fixed-budget models flag k rows whether or not anything is wrong, exactly
    like the paper's scoped forest, so compare against the "Model only" row of
    Table II and the feature-scope rows of Table IV, not against the review tier.
  * Run the 'full' (all features) and 'core6' variants to test whether PyOD
    models show the same feature-dilution effect as Isolation Forest.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bench_common import (get_feature_fn, load_ground_truth, load_manifest,
                          read_table, score_flags, summarize)


def build_autoencoder(seed):
    from pyod.models.auto_encoder import AutoEncoder
    # parameter names differ between pyod 1.x and 2.x; try known variants
    for kw in (
        {"epoch_num": 10, "batch_size": 512, "random_state": seed},
        {"epochs": 10, "batch_size": 512, "random_state": seed},
        {"epochs": 10, "batch_size": 512},
        {},
    ):
        try:
            return AutoEncoder(**kw)
        except TypeError:
            continue
    raise RuntimeError("Could not construct pyod AutoEncoder")


def make_models(seed, wanted):
    factories = {}

    def reg(name, fn):
        if wanted is None or name in wanted:
            factories[name] = fn

    try:
        from pyod.models.iforest import IForest
        reg("IForest", lambda: IForest(n_estimators=100, random_state=seed, n_jobs=-1))
    except ImportError:
        warnings.warn("pyod IForest unavailable")
    try:
        from pyod.models.ecod import ECOD
        reg("ECOD", lambda: ECOD())
    except ImportError:
        warnings.warn("pyod ECOD unavailable")
    try:
        from pyod.models.copod import COPOD
        reg("COPOD", lambda: COPOD())
    except ImportError:
        warnings.warn("pyod COPOD unavailable")
    try:
        from pyod.models.hbos import HBOS
        reg("HBOS", lambda: HBOS())
    except ImportError:
        warnings.warn("pyod HBOS unavailable")
    try:
        from pyod.models.feature_bagging import FeatureBagging
        from pyod.models.iforest import IForest as _IF
        # IForest as the base detector: the default LOF is infeasible on ~1M rows.
        # This is the closest PyOD analogue of the scoped ensemble (feature subsets
        # chosen at random rather than by source field). If your pyod version names
        # the argument differently this model is skipped with a warning.
        reg("FeatureBagging", lambda: FeatureBagging(
            base_estimator=_IF(n_estimators=50, random_state=seed),
            n_estimators=10, random_state=seed))
    except ImportError:
        warnings.warn("pyod FeatureBagging unavailable")
    try:
        reg("AutoEncoder", lambda: build_autoencoder(seed))
    except ImportError:
        warnings.warn("pyod AutoEncoder unavailable")
    return factories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="pyod_results.csv")
    ap.add_argument("--variants", nargs="+", default=["full", "core6"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.005])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--fit-sample", type=int, default=200_000)
    ap.add_argument("--features-fn", default=None,
                    help="module:function(trans_df, variant) -> (X, ids) to use the paper's exact matrix")
    args = ap.parse_args()

    feats_fn = get_feature_fn(args.features_fn)
    rows = []
    for run in load_manifest(args.manifest):
        seed = run["seed"]
        trans = read_table(run["data"], "trans")
        truth = load_ground_truth(run["faults"], "trans")
        for variant in args.variants:
            X, ids = feats_fn(trans, variant)
            Xs = np.nan_to_num(StandardScaler().fit_transform(X.values.astype("float32")))
            rng = np.random.default_rng(seed)
            fit_idx = rng.choice(len(Xs), size=min(args.fit_sample, len(Xs)), replace=False)
            for name, factory in make_models(seed, args.models).items():
                try:
                    t0 = time.time()
                    model = factory()
                    model.fit(Xs[fit_idx])
                    t1 = time.time()
                    scores = model.decision_function(Xs)  # higher = more anomalous
                    t2 = time.time()
                except Exception as exc:  # keep going if one model/version fails
                    warnings.warn(f"{name}/{variant}/seed{seed} failed: {exc}")
                    continue
                order = np.argsort(-scores)
                for b in args.budgets:
                    k = int(round(b * len(Xs))) if b < 1 else int(b)
                    flagged = ids.values[order[:k]]
                    rows.append({"seed": seed, "variant": variant, "model": name,
                                 "budget": b, "n_features": Xs.shape[1],
                                 "fit_s": t1 - t0, "score_s": t2 - t1,
                                 **score_flags(flagged, truth)})
                print(f"seed {seed} | {variant:6s} | {name:14s} done ({t2 - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    summ = summarize(res, ["variant", "model", "budget"])
    summ.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ[["variant", "model", "budget", "flags_mean", "precision_lb_mean",
                "precision_lb_sd", "recall_mean", "recall_sd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
