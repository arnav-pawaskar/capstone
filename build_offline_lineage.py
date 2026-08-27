"""
Offline lineage graph for the Berka / PKDD'99 dataset.

Replaces the Neo4j server with two embedded, file-based artefacts:

  LAYER 1  metadata lineage graph  -> NetworkX DiGraph, ~8 nodes, saved as GraphML
           This is what answers "which upstream table caused the bad data".
  LAYER 2  record-level graph      -> Kuzu embedded DB, ~1.07M nodes, single file
           This is what answers "which accounts/districts do the bad rows cluster in".

No server, no ports, no cloud. Both artefacts are files you can commit or zip.

Usage:
    python build_offline_lineage.py --raw ./data/raw --out ./artifacts
"""

import argparse
import os
import time

import kuzu
import networkx as nx
import pandas as pd

# ----------------------------------------------------------------------------
# Foreign-key topology, transcribed from the PKDD'99 ER diagram.
# (child_table, child_column) -> (parent_table, parent_column)
# ----------------------------------------------------------------------------
FOREIGN_KEYS = [
    ("account", "district_id", "district", "district_id"),
    ("client",  "district_id", "district", "district_id"),
    ("disp",    "client_id",   "client",   "client_id"),
    ("disp",    "account_id",  "account",  "account_id"),
    ("card",    "disp_id",     "disp",     "disp_id"),
    ("loan",    "account_id",  "account",  "account_id"),
    ("order",   "account_id",  "account",  "account_id"),
    ("trans",   "account_id",  "account",  "account_id"),
]

DISTRICT_COLS = [
    "district_id", "name", "region", "pop", "m499", "m1999", "m9999", "m10000",
    "n_cities", "urban_ratio", "avg_salary", "unemp95", "unemp96",
    "entrep_per1000", "crime95", "crime96",
]


def load_raw(raw_dir):
    """Read the eight semicolon-delimited CSVs into DataFrames."""
    tables = {}
    for name in ["account", "card", "client", "disp", "district",
                 "loan", "order", "trans"]:
        df = pd.read_csv(os.path.join(raw_dir, f"{name}.csv"),
                         sep=";", low_memory=False)
        if name == "district":
            # district.csv ships with opaque A1..A16 headers; rename to real names
            df.columns = DISTRICT_COLS
        # pandas 3.x emits the new 'str' dtype, which kuzu 0.11 cannot map.
        # Colab's pandas 2.2.3 uses object and is unaffected; this is a no-op there.
        for col in df.columns:
            if str(df[col].dtype) in ("str", "string"):
                df[col] = df[col].astype(object)
        tables[name] = df
    return tables


# ----------------------------------------------------------------------------
# LAYER 1 — metadata lineage graph (NetworkX)
# ----------------------------------------------------------------------------
def build_metadata_graph(tables, out_dir):
    """Table-level lineage DAG. Edge direction = child depends on parent."""
    g = nx.DiGraph()
    for name, df in tables.items():
        # 'label' is the attribute Gephi reads for on-screen node text.
        g.add_node(name, label=f"{name} ({len(df):,})", rows=len(df),
                   columns=",".join(map(str, df.columns)))

    for child, child_col, parent, parent_col in FOREIGN_KEYS:
        orphans = int((~tables[child][child_col].isin(
            tables[parent][parent_col])).sum())
        g.add_edge(child, parent, label=f"{child_col}->{parent_col}",
                   fk=f"{child_col}->{parent_col}", orphan_rows=orphans,
                   weight=1.0)

    path = os.path.join(out_dir, "lineage_metadata.graphml")
    nx.write_graphml(g, path)
    print(f"[layer 1] {g.number_of_nodes()} tables, {g.number_of_edges()} FK edges "
          f"-> {path}")
    return g


def trace_root_cause(g, table):
    """All upstream tables a fault in `table` could have propagated from."""
    return nx.descendants(g, table)


def blast_radius(g, table):
    """All downstream tables affected if `table` is corrupted."""
    return nx.ancestors(g, table)


# ----------------------------------------------------------------------------
# LAYER 2 — record-level graph (Kuzu, embedded)
# ----------------------------------------------------------------------------
DDL = """
CREATE NODE TABLE District(district_id INT64, name STRING, region STRING, avg_salary INT64, PRIMARY KEY(district_id));
CREATE NODE TABLE Client(client_id INT64, birth_number INT64, PRIMARY KEY(client_id));
CREATE NODE TABLE Account(account_id INT64, frequency STRING, date INT64, PRIMARY KEY(account_id));
CREATE NODE TABLE Disp(disp_id INT64, type STRING, PRIMARY KEY(disp_id));
CREATE NODE TABLE Card(card_id INT64, type STRING, PRIMARY KEY(card_id));
CREATE NODE TABLE Loan(loan_id INT64, amount DOUBLE, duration INT64, status STRING, PRIMARY KEY(loan_id));
CREATE NODE TABLE Ord(order_id INT64, amount DOUBLE, k_symbol STRING, PRIMARY KEY(order_id));
CREATE NODE TABLE Trans(trans_id INT64, date INT64, type STRING, operation STRING, amount DOUBLE, balance DOUBLE, PRIMARY KEY(trans_id));
CREATE REL TABLE ACCOUNT_IN_DISTRICT(FROM Account TO District);
CREATE REL TABLE CLIENT_IN_DISTRICT(FROM Client TO District);
CREATE REL TABLE DISP_OF_CLIENT(FROM Disp TO Client);
CREATE REL TABLE DISP_ON_ACCOUNT(FROM Disp TO Account);
CREATE REL TABLE CARD_FOR_DISP(FROM Card TO Disp);
CREATE REL TABLE LOAN_ON_ACCOUNT(FROM Loan TO Account);
CREATE REL TABLE ORDER_ON_ACCOUNT(FROM Ord TO Account);
CREATE REL TABLE TRANS_ON_ACCOUNT(FROM Trans TO Account);
"""


def build_record_graph(tables, out_dir):
    db_path = os.path.join(out_dir, "lineage_db")
    for suffix in ("", ".wal", ".tmp"):
        if os.path.exists(db_path + suffix):
            os.system(f"rm -rf '{db_path + suffix}'")

    conn = kuzu.Connection(kuzu.Database(db_path))
    for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
        conn.execute(stmt)

    t0 = time.time()
    node_specs = [
        ("District", tables["district"][["district_id", "name", "region", "avg_salary"]]),
        ("Client",   tables["client"][["client_id", "birth_number"]]),
        ("Account",  tables["account"][["account_id", "frequency", "date"]]),
        ("Disp",     tables["disp"][["disp_id", "type"]]),
        ("Card",     tables["card"][["card_id", "type"]]),
        ("Loan",     tables["loan"][["loan_id", "amount", "duration", "status"]]),
        ("Ord",      tables["order"][["order_id", "amount", "k_symbol"]]),
        ("Trans",    tables["trans"][["trans_id", "date", "type", "operation",
                                      "amount", "balance"]]),
    ]
    for label, df in node_specs:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype.name.startswith(("float", "Float")):
                df[col] = df[col].astype("float64")      # kuzu needs real float64
            elif df[col].dtype == object:
                df[col] = df[col].fillna("")             # NaN in a STRING col crashes kuzu
        conn.execute(f"COPY {label} FROM $df", {"df": df})

    rel_specs = [
        ("ACCOUNT_IN_DISTRICT", tables["account"][["account_id", "district_id"]]),
        ("CLIENT_IN_DISTRICT",  tables["client"][["client_id", "district_id"]]),
        ("DISP_OF_CLIENT",      tables["disp"][["disp_id", "client_id"]]),
        ("DISP_ON_ACCOUNT",     tables["disp"][["disp_id", "account_id"]]),
        ("CARD_FOR_DISP",       tables["card"][["card_id", "disp_id"]]),
        ("LOAN_ON_ACCOUNT",     tables["loan"][["loan_id", "account_id"]]),
        ("ORDER_ON_ACCOUNT",    tables["order"][["order_id", "account_id"]]),
        ("TRANS_ON_ACCOUNT",    tables["trans"][["trans_id", "account_id"]]),
    ]
    for label, df in rel_specs:
        conn.execute(f"COPY {label} FROM $df", {"df": df})

    size_mb = os.path.getsize(db_path) / 1e6
    print(f"[layer 2] record graph built in {time.time() - t0:.1f}s "
          f"-> {db_path} ({size_mb:.0f} MB, single file)")
    return conn


# ----------------------------------------------------------------------------
def demo(g, conn):
    print("\n--- layer 1: table-level lineage ---")
    print("  upstream of trans :", sorted(trace_root_cause(g, "trans")))
    print("  blast radius of district:", sorted(blast_radius(g, "district")))
    print("  FK violations:", {f"{u}->{v}": d["orphan_rows"]
                               for u, v, d in g.edges(data=True)})

    print("\n--- layer 2: record-level root cause (Cypher, 3 hops) ---")
    print(conn.execute("""
        MATCH (t:Trans)-[:TRANS_ON_ACCOUNT]->(a:Account)-[:ACCOUNT_IN_DISTRICT]->(d:District)
        WHERE t.balance < 0
        RETURN d.name AS district, d.region AS region, count(t) AS bad_rows
        ORDER BY bad_rows DESC LIMIT 5
    """).get_as_df().to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="./data/raw", help="dir holding the 8 raw CSVs")
    p.add_argument("--out", default="./artifacts", help="output dir for graph files")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tables = load_raw(args.raw)
    meta = build_metadata_graph(tables, args.out)
    conn = build_record_graph(tables, args.out)
    demo(meta, conn)