"""
Score a detector's anomalies.json against inject_faults2.py's injected_faults.json.

Berka ships no labels, so detect_anomalies.py validates by triangulation (loan-status
lift, agreement with a balance rule). Fault injection supplies the labels that were
missing -- but nothing consumed them, so detection accuracy on the injected faults had
never actually been measured. This closes that loop.

Recall is reported per fault type, because the number that matters is not one global
percentage: the fault classes have completely different detectability, and averaging
them hides which half of the system is doing the work.

Key-space caveat, handled here so it cannot silently zero out a whole class: for point
faults the ground truth's `record_id` is the corrupted row's own primary key, but for
volume_spikes it is the *parent* the burst hung off (an account_id / district_id /
disp_id). The flaggable rows are the synthetic children, listed in `record_ids`. Older
ground-truth files predate that field, so their ids are recovered from the detail text.

Usage:
    python evaluate_detection.py --faults ./data/faulty/injected_faults.json \
                                 --anomalies ./artifacts_faulty/anomalies.json
"""

import argparse
import collections
import json
import re

# Fallback for ground-truth files written before `record_ids` existed.
SPIKE_IDS = re.compile(r"(?:trans|loan|card)_ids (\d+)-(\d+)")


def load_ground_truth(path):
    """-> {(table, record_id): {fault_type, ...}}"""
    gt = collections.defaultdict(set)
    for f in json.load(open(path)):
        table, ftype = f["table"], f["fault_type"]
        ids = f.get("record_ids")
        if ids is None and ftype == "volume_spikes":
            m = SPIKE_IDS.search(f.get("detail", ""))
            ids = range(int(m.group(1)), int(m.group(2)) + 1) if m else None
        for rid in (ids if ids is not None else [f["record_id"]]):
            gt[(table, int(rid))].add(ftype)
    return gt


def load_flagged(path):
    """-> {(table, record_id)}"""
    return {(a["dataset"].removesuffix(".csv"), int(a["record_id"]))
            for a in json.load(open(path))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--faults", default="./data/faulty/injected_faults.json")
    p.add_argument("--anomalies", default="./artifacts_faulty/anomalies.json")
    p.add_argument("--by-table", action="store_true",
                   help="also break each fault type down by table")
    args = p.parse_args()

    gt = load_ground_truth(args.faults)
    flagged = load_flagged(args.anomalies)

    # A record can carry more than one fault type; it counts once per type, so
    # the per-type denominators sum to more than the number of corrupted records.
    by_type = collections.defaultdict(lambda: [0, 0])
    by_table = collections.defaultdict(lambda: [0, 0])
    for key, ftypes in gt.items():
        hit = key in flagged
        for ft in ftypes:
            by_type[ft][0] += hit
            by_type[ft][1] += 1
            by_table[(ft, key[0])][0] += hit
            by_table[(ft, key[0])][1] += 1

    print(f"=== recall by fault type ===")
    print(f"  {len(gt):,} corrupted records, {len(flagged):,} records flagged\n")
    print(f"  {'fault_type':24s} {'caught':>8s} {'total':>8s} {'recall':>8s}")
    for ft, (c, n) in sorted(by_type.items()):
        print(f"  {ft:24s} {c:8,} {n:8,} {c / n:8.1%}")
    tc = sum(v[0] for v in by_type.values())
    tn = sum(v[1] for v in by_type.values())
    print(f"  {'-' * 50}")
    print(f"  {'ALL':24s} {tc:8,} {tn:8,} {tc / max(tn, 1):8.1%}")

    if args.by_table:
        print("\n=== recall by fault type x table ===")
        for (ft, t), (c, n) in sorted(by_table.items()):
            print(f"  {ft:24s} {t:9s} {c:6,}/{n:<6,} {c / n:7.1%}")

    caught = sum(1 for k in gt if k in flagged)
    print("\n=== precision ===")
    print(f"  flagged and injected        : {caught:,} / {len(flagged):,} "
          f"= {caught / max(len(flagged), 1):.1%}")
    print("  Berka contains genuine outliers that were never injected, so this is a"
          "\n  lower bound on precision, not an error rate.")


if __name__ == "__main__":
    main()
