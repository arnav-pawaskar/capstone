"""
Render a Kuzu query result as a self-contained, fully offline HTML graph.

Why not pyvis: pyvis's `cdn_resources='local'` is broken — the generated HTML
still contains <script src="https://cdnjs.cloudflare.com/...">. On a machine
with no internet the graph renders blank. That is exactly the failure you do
NOT want during a viva. This script inlines the JS instead, so the output is
one HTML file with zero network calls.

One-time setup (needs internet ONCE, then commit the .js to your repo):
    pip install pyvis
    cp $(python -c "import pyvis,os;print(os.path.dirname(pyvis.__file__))")/templates/lib/vis-9.1.2/vis-network.min.js assets/
    # (or download vis-network.min.js from the vis.js releases page)

Usage:
    python render_offline_graph.py --db artifacts/lineage_db \
        --query "MATCH p=(t:Trans)-[:TRANS_ON_ACCOUNT]->(a:Account) WHERE t.balance < 0 RETURN * LIMIT 50" \
        --out artifacts/subgraph.html
"""

import argparse
import json
import os

import kuzu

PALETTE = {
    "Trans": "#e8845f", "Account": "#5b8ff9", "District": "#61ddaa",
    "Client": "#f6bd16", "Disp": "#9270ca", "Card": "#78d3f8",
    "Loan": "#ff9d4d", "Ord": "#b6e3f4",
}

# Label -> the property to show on hover, so a node is identifiable in the viva
KEY_PROP = {
    "Trans": "trans_id", "Account": "account_id", "District": "name",
    "Client": "client_id", "Disp": "disp_id", "Card": "card_id",
    "Loan": "loan_id", "Ord": "order_id",
}

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
{css}
body {{ font-family: system-ui, sans-serif; margin: 0; padding: 16px; }}
#graph {{ height: 720px; border: 1px solid #ddd; border-radius: 6px; }}
#legend {{ margin: 12px 0; font-size: 13px; }}
.chip {{ display:inline-block; padding:3px 10px; margin-right:6px;
         border-radius:12px; color:#fff; }}
h2 {{ margin: 0 0 4px; font-size: 16px; }}
code {{ background:#f4f4f4; padding:2px 6px; border-radius:3px; font-size:12px; }}
</style>
<script>{js}</script></head>
<body>
<h2>{title}</h2>
<div><code>{query}</code></div>
<div id="legend">{legend}</div>
<div id="graph"></div>
<script>
new vis.Network(
  document.getElementById("graph"),
  {{ nodes: new vis.DataSet({nodes}), edges: new vis.DataSet({edges}) }},
  {{ physics: {{ stabilization: true, barnesHut: {{ springLength: 120 }} }},
     edges: {{ color: "#aaa", smooth: {{ type: "continuous" }} }},
     nodes: {{ shape: "dot", size: 14, font: {{ size: 12 }} }} }}
);
</script></body></html>"""


def render(db_path, query, out_path, js_path, css_path=None, title="Lineage subgraph"):
    conn = kuzu.Connection(kuzu.Database(db_path))
    g = conn.execute(query).get_as_networkx()

    nodes, labels_seen = [], set()
    for node_id, data in g.nodes(data=True):
        label = data.get("_label", "Node")
        labels_seen.add(label)
        key = KEY_PROP.get(label)
        caption = f"{label} {data.get(key)}" if key and data.get(key) is not None else label
        nodes.append({
            "id": node_id,
            "label": label,
            "title": caption,
            "color": PALETTE.get(label, "#999999"),
        })
    edges = [{"from": u, "to": v, "arrows": "to"} for u, v in g.edges()]

    legend = "".join(
        f'<span class="chip" style="background:{PALETTE.get(l, "#999")}">{l}</span>'
        for l in sorted(labels_seen)
    )

    html = TEMPLATE.format(
        title=title,
        query=query.strip().replace("<", "&lt;").replace(">", "&gt;"),
        css=open(css_path).read() if css_path and os.path.exists(css_path) else "",
        js=open(js_path).read(),
        legend=legend,
        nodes=json.dumps(nodes),
        edges=json.dumps(edges),
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
          f"-> {out_path} ({os.path.getsize(out_path) // 1024} KB, self-contained)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="artifacts/lineage_db")
    p.add_argument("--query", required=True, help="Cypher; must RETURN whole nodes (RETURN * or RETURN a,b)")
    p.add_argument("--out", default="artifacts/subgraph.html")
    p.add_argument("--js", default="assets/vis-network.min.js")
    p.add_argument("--css", default="assets/vis-network.css")
    p.add_argument("--title", default="Lineage subgraph")
    a = p.parse_args()
    render(a.db, a.query, a.out, a.js, a.css, a.title)