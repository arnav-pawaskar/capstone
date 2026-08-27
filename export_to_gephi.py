"""
Export a Kuzu query result to GraphML so it opens in Gephi with readable labels.

Gephi is fully offline, which suits the constraint — but Gephi has no query
language. It filters and lays out a graph you hand it; it cannot answer a
Cypher question. So the division of labour is:

    Kuzu   -> stores the dataset, runs the queries        (the "queried" half)
    Gephi  -> renders whatever a query returned           (the "on a graph" half)

Do NOT try to load all 1.06M Trans nodes into Gephi. It will technically open
and then be unusable. Export the subgraph a query returned — a few hundred to a
few thousand nodes is where Gephi is actually good.

Usage:
    python export_to_gephi.py \
      --query "MATCH (t:Trans)-[:TRANS_ON_ACCOUNT]->(a:Account)-[:ACCOUNT_IN_DISTRICT]->(d:District) WHERE t.balance < 0 RETURN * LIMIT 300" \
      --out artifacts/bad_balance_subgraph.graphml
"""

import argparse
import os

import kuzu
import networkx as nx

KEY_PROP = {
    "Trans": "trans_id", "Account": "account_id", "District": "name",
    "Client": "client_id", "Disp": "disp_id", "Card": "card_id",
    "Loan": "loan_id", "Ord": "order_id",
}


def export(db_path, query, out_path, label_types=None):
    conn = kuzu.Connection(kuzu.Database(db_path))
    src = conn.execute(query).get_as_networkx()

    # Rebuild as a clean DiGraph: kuzu's export gives every node every column
    # (mostly None), and GraphML cannot serialise None values.
    g = nx.DiGraph()
    for node_id, data in src.nodes(data=True):
        label = data.get("_label", "Node")
        key = KEY_PROP.get(label)
        val = data.get(key)
        # Only label the node types worth labelling. Labelling 60 Trans leaves
        # turns a clean figure into noise; District/Account carry the story.
        show = label_types is None or label in label_types
        attrs = {
            # Zero-width space (U+200B), NOT "" and NOT " ": Gephi falls back to
            # rendering the node Id whenever the Label is empty, and it trims
            # whitespace-only labels to empty first. U+200B is not ASCII
            # whitespace, so it survives the trim and renders as nothing.
            "label": (f"{label} {val}" if val is not None else label) if show else "\u200b",
            "node_type": label,  # partition on this in Gephi to colour by node type
        }
        for k, v in data.items():
            if v is not None and not k.startswith("_"):
                attrs[k] = v
        g.add_node(str(node_id), **attrs)

    for u, v in src.edges():
        g.add_edge(str(u), str(v), weight=1.0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nx.write_graphml(g, out_path)
    print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {out_path}")
    print("In Gephi: Appearance > Nodes > Partition > 'node_type' > Apply, "
          "then Layout > ForceAtlas 2 > Run.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="artifacts/lineage_db")
    p.add_argument("--query", required=True)
    p.add_argument("--out", default="artifacts/subgraph.graphml")
    p.add_argument("--label-types", default=None,
                   help="comma-separated node types to label, e.g. District,Account. "
                        "Omit to label everything.")
    a = p.parse_args()
    export(a.db, a.query, a.out,
           set(a.label_types.split(",")) if a.label_types else None)