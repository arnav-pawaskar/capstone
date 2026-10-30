"""
Score FROZEN detector output against the blind ground-truth log.

Run the unchanged detectors on the blind copy (out-dir of blind_injector.py)
first; each run produces an anomalies.json of {dataset, record_id, ...}. Pass one
file per tier/detector so tiers can be reported separately:

    python evaluate_blind.py --faults sealed/blind_faults.json \
        --detector review=out/review_anomalies.json \
        --detector model=out/residual_anomalies.json \
        --clean-flags review=clean/review_anomalies.json \
        --clean-flags model=clean/residual_anomalies.json \
        --out blind_results.csv

Reported per fault class and detector:
  recall_strict   injected row itself flagged
  recall_lenient  injected row OR its source row flagged (matters for replayed_txn,
                  where a detector may flag the original instead of the replay)
  window_recall   for window faults: share of windows with >=1 flagged row
Per detector overall: flags, precision lower bound, precision excluding flags the
same detector already raises on the clean copy (--clean-flags label=path, one per
detector, same labels as --detector), and overall recall.
A union row ("ALL") is added when more than one detector is given.
"""
import argparse
import json
from collections import defaultdict

import pandas as pd


def load_flags(path):
    with open(path) as f:
        return {(a["dataset"].replace(".csv", ""), int(a["record_id"])) for a in json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", required=True)
    ap.add_argument("--detector", action="append", required=True, help="label=path/to/anomalies.json")
    ap.add_argument("--clean-flags", action="append", default=[], help="label=path/to/clean_anomalies.json")
    ap.add_argument("--out", default="blind_results.csv")
    args = ap.parse_args()

    faults = json.load(open(args.faults))
    truth_keys = {(f["table"], int(f["record_id"])) for f in faults}
    by_class = defaultdict(list)
    windows = defaultdict(lambda: defaultdict(set))
    for f in faults:
        by_class[f["fault_class"]].append(f)
        if "window_id" in f:
            windows[f["fault_class"]][f["window_id"]].add((f["table"], int(f["record_id"])))

    detectors = {}
    for spec in args.detector:
        label, path = spec.split("=", 1)
        detectors[label] = load_flags(path)
    if len(detectors) > 1:
        detectors["ALL"] = set().union(*detectors.values())
    clean_by = {}
    for spec in args.clean_flags:
        label, path = spec.split("=", 1)
        clean_by[label] = load_flags(path)
    if len(detectors) > 1 and clean_by:
        clean_by["ALL"] = set().union(*clean_by.values())

    class_rows, overall_rows = [], []
    for label, flagged in detectors.items():
        for cls, items in sorted(by_class.items()):
            strict = lenient = 0
            for f in items:
                key = (f["table"], int(f["record_id"]))
                hit = key in flagged
                strict += hit
                lenient += hit or any((f["table"], int(a)) in flagged for a in f.get("alt_ids", []))
            wins = windows.get(cls)
            class_rows.append({
                "detector": label, "fault_class": cls, "n_injected": len(items),
                "recall_strict": strict / len(items), "recall_lenient": lenient / len(items),
                "window_recall": (sum(bool(ids & flagged) for ids in wins.values()) / len(wins)) if wins else None,
            })
        hit = flagged & truth_keys
        new_flags = flagged - clean_by.get(label, set())
        overall_rows.append({
            "detector": label, "flags": len(flagged), "flags_not_on_clean": len(new_flags),
            "precision_lb": len(hit) / max(len(flagged), 1),
            "precision_lb_excl_clean": len(hit) / max(len(new_flags), 1),
            "recall": len(hit) / max(len(truth_keys), 1), "n_injected": len(truth_keys),
        })

    cls_df, all_df = pd.DataFrame(class_rows), pd.DataFrame(overall_rows)
    cls_df.to_csv(args.out, index=False)
    all_df.to_csv(args.out.replace(".csv", "_overall.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n== per class ==")
    print(cls_df.round(3).to_string(index=False))
    print("\n== overall ==")
    print(all_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
