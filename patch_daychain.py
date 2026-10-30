"""
Crash fix for anomaly_detection._day_chain_unresolved. Changes NO detection logic.

The function walks each (account, day) group and records whether the day's balance
arithmetic closed. It stored that result in a dict keyed by (account_id, date) and
later looked each row's key up again. That second lookup is the fragile step: it
needs the key built from the group index to compare/hash equal to the key built
from the raw row values, and it raised KeyError on the blind-injected data.

The fix keeps the walk exactly as is, but stores each group's result by position
and maps rows to groups with groupby.ngroup() (the same grouping, same order),
so no key comparison is involved. Rows whose group key is missing (NaN account or
unparseable date) get False ("not unresolved") instead of crashing.

Usage (from the folder containing anomaly_detection.py):
    python patch_daychain.py anomaly_detection.py
A backup is written to anomaly_detection.py.bak first. If any expected line is not
found exactly once the file is left untouched.
"""
import shutil
import sys

REPLACEMENTS = [
    (
        "    resolved = {}\n",
        "    resolved = np.zeros(len(net), dtype=bool)\n",
    ),
    (
        "    for key, acct, movement, balances in zip(net.index, accounts, net.values, seen.values):\n",
        "    for i, (acct, movement, balances) in enumerate(zip(accounts, net.values, seen.values)):\n",
    ),
    (
        "        resolved[key] = hit\n",
        "        resolved[i] = hit\n",
    ),
    (
        '    per_row = [not resolved[k] for k in zip(d["account_id"].values, d["_dt"].values)]\n',
        "    gid = grp.ngroup().to_numpy(dtype=float)  # group number per row, same grouping/order as `net`\n"
        "    ok = ~np.isnan(gid) & (gid >= 0)  # rows with a NaN key (no group) count as resolved\n"
        "    per_row = np.where(ok, ~resolved[np.where(ok, gid, 0).astype(int)], False)\n",
    ),
]


def main(path):
    with open(path, encoding="utf-8") as f:  # text mode: CRLF read as \n, written back per platform
        src = f.read()
    if "gid = grp.ngroup()" in src:
        print("already patched; nothing to do")
        return 0
    for old, _ in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            print(f"ABORT: expected exactly one match, found {n} for:\n{old}")
            return 1
    for old, new in REPLACEMENTS:
        src = src.replace(old, new)
    compile(src, path, "exec")  # refuse to write a file that no longer parses
    shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"patched {path} (backup: {path}.bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "anomaly_detection.py"))
