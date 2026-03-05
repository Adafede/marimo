# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "networkx==3.6.1",
#     "requests==2.32.5",
# ]
# ///

"""
Wikidata Connector

Copyright (C) 2026 Adriano Rutz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="Wikidata Entity Connector")

with app.setup:
    import json
    import marimo as mo
    import networkx as nx
    import requests

    QLEVER_ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"

    PREFIXES = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    """

    WDT_PREFIX = "http://www.wikidata.org/prop/direct/"
    WD_Q_PREFIX = "http://www.wikidata.org/entity/Q"

    # ── defaults ──────────────────────────────────────────────────────────────

    DEFAULT_TERMINALS = """\
    Q57065344
    Q55188662
    Q59698595
    Q58834148
    Q25707408
    Q53581593
    Q56888857
    Q83472787
    Q91396708
    Q70040821
    Q60735170
    Q138561779
    Q56909720
    Q45292460
    Q137075905
    Q29052381
    Q41679712
    Q138561927
    Q43370332
    Q56885211
    Q43370334
    Q57321153
    Q39404671
    Q97455964
    Q118122846
    Q27863244
    Q47474705
    Q43744369
    Q85885610
    Q88899374
    Q102725915
    Q138562002
    Q88000623
    Q50726922
    Q125697113
    Q50420339
    Q114240185
    """

    DEFAULT_BLOCKLIST = """\
    Q5
    Q1650915
    Q6581072
    Q6581097
    """

    # Properties to exclude
    DEFAULT_EXCLUDED_PROPS = """\
    P407
    """

    # Wikidata brand colors
    WD_GREEN = "#339966"
    WD_RED = "#990000"
    WD_BLUE = "#006699"

    # ── core helpers ──────────────────────────────────────────────────────────

    def sparql(query: str, timeout: int = 60) -> list[dict]:
        """Execute SPARQL query via POST to avoid URL length limits."""
        r = requests.post(
            QLEVER_ENDPOINT,
            data={"query": PREFIXES + query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
        if not r.ok:
            raise RuntimeError(f"QLever {r.status_code}: {r.text[:400]}")
        data = r.json()
        vars_ = data["head"]["vars"]
        return [
            {v: row[v]["value"] for v in vars_ if v in row}
            for row in data["results"]["bindings"]
        ]

    def qid(uri: str) -> str:
        return uri.rsplit("/", 1)[-1]

    def fetch_labels_batch(entities: list[str]) -> dict[str, str]:
        """Fetch labels for multiple entities in a single query."""
        if not entities:
            return {}
        values = " ".join(f"wd:{e}" for e in entities)
        rows = sparql(f"""
            SELECT ?entity ?l WHERE {{
              VALUES ?entity {{ {values} }}
              ?entity rdfs:label ?l .
              FILTER(LANG(?l) = "en" || LANG(?l) = "mul")
            }}
        """)
        result = {qid(r["entity"]): r["l"] for r in rows}
        for e in entities:
            result.setdefault(e, e)
        return result

    def fetch_prop_labels_batch(pids: list[str]) -> dict[str, str]:
        """Fetch property labels in a single query."""
        if not pids:
            return {}
        values = " ".join(f"wd:{p}" for p in pids)
        rows = sparql(f"""
            SELECT ?prop ?l WHERE {{
              VALUES ?prop {{ {values} }}
              ?prop rdfs:label ?l .
              FILTER(LANG(?l) = "en" || LANG(?l) = "mul")
            }}
        """)
        result = {qid(r["prop"]): r["l"] for r in rows}
        for p in pids:
            result.setdefault(p, p)
        return result

    # ── main algorithm: multi-source BFS with single query per depth ─────────

    MAX_FRONTIER = 500  # Limit frontier size to avoid huge queries

    def connect_entities(
        terminals: list[str],
        max_depth: int,
        blocklist: set[str],
        excluded_props: set[str],
    ) -> nx.Graph:
        """
        Fast multi-source BFS: expand from ALL terminals at once.
        Two SPARQL queries per depth level (outgoing + incoming). Returns Steiner tree.
        """
        terminal_set = set(terminals)
        blocklist_set = set(blocklist)
        excluded_props_set = set(excluded_props)

        # Edges discovered: (a, b) -> property
        edges: dict[tuple[str, str], str] = {}

        # All reached nodes
        reached: set[str] = set(terminals)

        frontier = set(terminals)

        for depth in range(1, max_depth + 1):
            if not frontier:
                break

            # Limit frontier size
            if len(frontier) > MAX_FRONTIER:
                frontier = set(list(frontier)[:MAX_FRONTIER])

            values = " ".join(f"wd:{q}" for q in frontier)

            # Query 1: get all outgoing edges
            rows_out = sparql(
                f"""
                SELECT ?src ?p ?tgt WHERE {{
                  VALUES ?src {{ {values} }}
                  ?src ?p ?tgt .
                  FILTER(STRSTARTS(STR(?p), "{WDT_PREFIX}"))
                  FILTER(STRSTARTS(STR(?tgt), "{WD_Q_PREFIX}"))
                }}
            """,
                timeout=90,
            )

            # Query 2: get all incoming edges
            rows_in = sparql(
                f"""
                SELECT ?src ?p ?tgt WHERE {{
                  VALUES ?tgt {{ {values} }}
                  ?src ?p ?tgt .
                  FILTER(STRSTARTS(STR(?p), "{WDT_PREFIX}"))
                  FILTER(STRSTARTS(STR(?src), "{WD_Q_PREFIX}"))
                }}
            """,
                timeout=90,
            )

            new_frontier: set[str] = set()

            # Process outgoing edges
            for row in rows_out:
                src, tgt, prop = qid(row["src"]), qid(row["tgt"]), qid(row["p"])
                if prop in excluded_props_set or tgt in blocklist_set:
                    continue
                edge_key = (min(src, tgt), max(src, tgt))
                if edge_key not in edges:
                    edges[edge_key] = prop
                if tgt not in reached:
                    reached.add(tgt)
                    new_frontier.add(tgt)

            # Process incoming edges
            for row in rows_in:
                src, tgt, prop = qid(row["src"]), qid(row["tgt"]), qid(row["p"])
                if prop in excluded_props_set or src in blocklist_set:
                    continue
                edge_key = (min(src, tgt), max(src, tgt))
                if edge_key not in edges:
                    edges[edge_key] = prop
                if src not in reached:
                    reached.add(src)
                    new_frontier.add(src)

            frontier = new_frontier

        # Build graph from discovered edges
        edge_graph = nx.Graph()
        for (a, b), prop in edges.items():
            edge_graph.add_edge(a, b, prop=prop)

        # Find which terminals are in the graph
        reachable_terminals = terminal_set & set(edge_graph.nodes())

        if len(reachable_terminals) < 2:
            G = nx.Graph()
            for t in terminals:
                G.add_node(t)
            return G

        # Compute Steiner tree: shortest paths between terminal pairs, then MST
        terminal_list = list(reachable_terminals)
        path_graph = nx.Graph()

        for i, t1 in enumerate(terminal_list):
            for t2 in terminal_list[i + 1 :]:
                try:
                    path = nx.shortest_path(edge_graph, t1, t2)
                    path_graph.add_edge(t1, t2, weight=len(path) - 1, path=path)
                except nx.NetworkXNoPath:
                    pass

        if path_graph.number_of_edges() == 0:
            G = nx.Graph()
            for t in terminals:
                G.add_node(t)
            return G

        # Handle disconnected path_graph
        if not nx.is_connected(path_graph):
            largest_cc = max(nx.connected_components(path_graph), key=len)
            path_graph = path_graph.subgraph(largest_cc).copy()

        # Get MST and build final graph
        mst = nx.minimum_spanning_tree(path_graph, weight="weight")
        G = nx.Graph()

        for t1, t2, data in mst.edges(data=True):
            path = data.get("path", [t1, t2])
            for a, b in zip(path, path[1:]):
                if edge_graph.has_edge(a, b):
                    G.add_edge(a, b, prop=edge_graph[a][b].get("prop", ""))

        for t in terminals:
            G.add_node(t)

        return G

    # ── vis.js graph HTML ─────────────────────────────────────────────────────

    def build_graph_html(G: nx.Graph, terminals: list[str]) -> str:
        # Batch fetch all labels at once
        node_labels = fetch_labels_batch(list(G.nodes()))
        prop_ids = list(
            {d.get("prop", "") for _, _, d in G.edges(data=True) if d.get("prop")},
        )
        edge_label_map = fetch_prop_labels_batch(prop_ids)

        vis_nodes = []
        for node in G.nodes():
            is_terminal = node in terminals
            lbl = node_labels.get(node, node)
            vis_nodes.append(
                {
                    "id": node,
                    "label": lbl,
                    "popupHtml": f"<b>{lbl}</b><br/><span style='color:#54595d;font-size:11px'>{node}</span><br/><a href='https://www.wikidata.org/wiki/{node}' target='_blank' style='color:#006699'>Open in Wikidata ↗</a>",
                    "group": "terminal" if is_terminal else "connector",
                    "shape": "ellipse" if is_terminal else "box",
                },
            )

        vis_edges = []
        for i, (src, tgt, data) in enumerate(G.edges(data=True)):
            pid = data.get("prop", "")
            plabel = edge_label_map.get(pid, pid)
            vis_edges.append(
                {
                    "id": i,
                    "from": src,
                    "to": tgt,
                    "label": plabel,
                    "popupHtml": f"<b>{plabel}</b><br/><span style='color:#54595d;font-size:11px'>{pid}</span><br/><a href='https://www.wikidata.org/wiki/{pid}' target='_blank' style='color:#006699'>Open in Wikidata ↗</a>",
                },
            )

        nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
        edges_json = json.dumps(vis_edges, ensure_ascii=False)

        inner_html = (
            """<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></scr"""
            + """ipt>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css"/>
    <style>
      html, body { margin: 0; padding: 0; height: 100%; background: #f8f9fa; font-family: 'Linux Libertine', 'Georgia', serif; }
      #g { width: 100%; height: 100%; }
      #popup {
    display: none;
    position: absolute;
    z-index: 999;
    background: #ffffff;
    border: 1px solid #a2a9b1;
    border-left: 3px solid #990000;
    border-radius: 2px;
    padding: 10px 14px;
    font-size: 13px;
    font-family: sans-serif;
    color: #202122;
    max-width: 260px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    line-height: 1.5;
      }
      #popup .close-btn {
    position: absolute; top: 4px; right: 8px;
    cursor: pointer; color: #54595d; font-size: 15px; line-height: 1;
      }
      #popup .close-btn:hover { color: #202122; }
    </style>
    </head>
    <body>
    <div id="g"></div>
    <div id="popup">
      <span class="close-btn" onclick="document.getElementById('popup').style.display='none'">✕</span>
      <div id="popup-content"></div>
    </div>
    <script>
      const nodesData = """
            + nodes_json
            + """;
      const edgesData = """
            + edges_json
            + """;

      const nodes = new vis.DataSet(nodesData);
      const edges = new vis.DataSet(edgesData);

      const network = new vis.Network(document.getElementById("g"), { nodes, edges }, {
    nodes: {
      font: { face: "Linux Libertine, Georgia, serif", size: 14, color: "#202122" },
      borderWidth: 1.5,
      borderWidthSelected: 2.5,
      shadow: { enabled: true, color: "rgba(0,0,0,0.10)", size: 6, x: 1, y: 2 },
    },
    edges: {
      font: { face: "sans-serif", size: 11, color: "#54595d", align: "middle", strokeWidth: 0, background: "#f8f9fa" },
      color: { color: "#990000", highlight: "#cc0000", hover: "#cc0000", inherit: false },
      width: 1.5,
      selectionWidth: 2.5,
      smooth: { type: "curvedCW", roundness: 0.18 },
      arrows: { to: { enabled: true, scaleFactor: 0.65, type: "arrow" } },
    },
    groups: {
      terminal: {
        color: {
          background: "#eaf3fb",
          border: "#006699",
          highlight: { background: "#cce0f0", border: "#006699" },
          hover: { background: "#d8ecf8", border: "#006699" },
        },
        font: { color: "#006699", size: 14 },
        shape: "ellipse",
        shadow: { enabled: true, color: "rgba(0,102,153,0.15)", size: 8 },
      },
      connector: {
        color: {
          background: "#edf7f1",
          border: "#339966",
          highlight: { background: "#d4efe3", border: "#2d8057" },
          hover: { background: "#d4efe3", border: "#2d8057" },
        },
        font: { color: "#2d8057", size: 12 },
        shape: "box",
      },
    },
    physics: {
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -80,
        centralGravity: 0.006,
        springLength: 200,
        springConstant: 0.05,
        damping: 0.45,
      },
      stabilization: { iterations: 250, fit: true },
    },
    interaction: {
      hover: true,
      tooltipDelay: 99999,
      hideEdgesOnDrag: false,
      zoomView: true,
      selectConnectedEdges: true,
    },
      });

      const nodeMap = {};
      nodesData.forEach(n => { nodeMap[n.id] = n; });
      const edgeMap = {};
      edgesData.forEach(e => { edgeMap[e.id] = e; });

      const popup = document.getElementById("popup");
      const popupContent = document.getElementById("popup-content");

      function showPopup(html, x, y) {
    popupContent.innerHTML = html;
    popup.style.display = "block";
    const pw = popup.offsetWidth || 260;
    const ph = popup.offsetHeight || 100;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let left = x + 12;
    let top = y + 12;
    if (left + pw > vw - 8) left = x - pw - 12;
    if (top + ph > vh - 8) top = y - ph - 12;
    popup.style.left = left + "px";
    popup.style.top = top + "px";
      }

      network.on("click", function(params) {
    const domPos = params.event.center || { x: params.event.pageX, y: params.event.pageY };
    if (params.nodes.length > 0) {
      const n = nodeMap[params.nodes[0]];
      if (n) showPopup(n.popupHtml, domPos.x, domPos.y);
    } else if (params.edges.length > 0) {
      const e = edgeMap[params.edges[0]];
      if (e) showPopup(e.popupHtml, domPos.x, domPos.y);
    } else {
      popup.style.display = "none";
    }
      });

      network.on("dragStart", () => { popup.style.display = "none"; });
      network.on("zoom", () => { popup.style.display = "none"; });
    </script>
    </body>
    </html>"""
        )

        srcdoc = inner_html.replace("&", "&amp;").replace('"', "&quot;")
        return f'<iframe srcdoc="{srcdoc}" style="width:100%;height:620px;border:1px solid #a2a9b1;border-radius:2px;display:block;"></iframe>'

    # ── SPARQL snippet ────────────────────────────────────────────────────────

    def build_sparql(G: nx.Graph) -> str:
        values_rows = "\n    ".join(
            f"(wd:{s} wdt:{d['prop']} wd:{t})"
            for s, t, d in G.edges(data=True)
            if "prop" in d
        )
        return f"""PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?s ?p ?o ?sLabel ?pLabel ?oLabel WHERE {{
      VALUES (?s ?p ?o) {{
    {values_rows}
      }}
      ?s rdfs:label ?sLabel . FILTER(LANG(?sLabel) = "en" || LANG(?sLabel) = "mul")
      BIND(IRI(REPLACE(STR(?p), "prop/direct", "entity")) AS ?pEntity)
      ?pEntity rdfs:label ?pLabel . FILTER(LANG(?pLabel) = "en" || LANG(?pLabel) = "mul")
      ?o rdfs:label ?oLabel . FILTER(LANG(?oLabel) = "en" || LANG(?oLabel) = "mul")
    }}"""


@app.cell
def header():
    mo.md("""
    # Wikidata Entity Connector

    Find the smallest subgraph connecting N Wikidata entities via **multi-source BFS** over the `wdt:` graph, using [QLever](https://qlever.cs.uni-freiburg.de/wikidata) as the SPARQL backend.
    """)
    return


@app.cell
def input_entities():
    entities_input = mo.ui.text_area(
        label="## Entities to connect (one QID per line)",
        placeholder="e.g., Q57065344",
        value=DEFAULT_TERMINALS,
        full_width=True,
    )
    entities_input
    return (entities_input,)


@app.cell
def input_blocklist():
    blocklist_input = mo.ui.text_area(
        label="## Blocklist (one QID per line — these nodes will never be used as connectors)",
        placeholder="e.g., Q5",
        value=DEFAULT_BLOCKLIST,
        full_width=True,
    )
    blocklist_input
    return (blocklist_input,)


@app.cell
def input_excluded_props():
    excluded_props_input = mo.ui.text_area(
        label="## Excluded properties (one PID per line — these properties will be ignored)",
        placeholder="e.g., P31",
        value=DEFAULT_EXCLUDED_PROPS,
        full_width=True,
    )
    excluded_props_input
    return (excluded_props_input,)


@app.cell
def input_params():
    depth_slider = mo.ui.slider(
        start=1,
        stop=6,
        value=3,
        step=1,
        label="Max BFS depth",
    )
    depth_slider
    return (depth_slider,)


@app.cell
def button_run():
    run_button = mo.ui.run_button(label="Connect entities")
    run_button
    return (run_button,)


@app.cell
def compute(
    blocklist_input,
    depth_slider,
    entities_input,
    excluded_props_input,
    run_button,
):
    mo.stop(not run_button.value)

    terminals = [
        q.strip()
        for q in entities_input.value.strip().splitlines()
        if q.strip().startswith("Q")
    ]
    blocklist = {
        q.strip()
        for q in blocklist_input.value.strip().splitlines()
        if q.strip().startswith("Q")
    }
    excluded_props = {
        p.strip()
        for p in excluded_props_input.value.strip().splitlines()
        if p.strip().startswith("P")
    }
    max_depth = depth_slider.value

    if len(terminals) < 2:
        graph_result = None
        error_msg = mo.callout(
            mo.md("[!] Please enter at least **2** QIDs."),
            kind="warn",
        )
    else:
        try:
            with mo.status.spinner(title="Running multi-source BFS via QLever…"):
                G = connect_entities(
                    terminals,
                    max_depth=max_depth,
                    blocklist=blocklist,
                    excluded_props=excluded_props,
                )
            graph_result = G
            error_msg = None
        except Exception as e:
            graph_result = None
            error_msg = mo.callout(mo.md(f"[!] **Error:** {e}"), kind="danger")
    return G, error_msg, graph_result, terminals


@app.cell
def show_error(error_msg):
    error_msg
    return


@app.cell
def render_graph(graph_result):
    mo.stop(graph_result is None)
    mo.md(f"""
    ## Result

    **{graph_result.number_of_nodes()} nodes**, **{graph_result.number_of_edges()} edges**

    <div style='display:flex;gap:20px;font-size:12px;font-family:sans-serif;margin-bottom:8px;align-items:center'>
      <span style='display:flex;align-items:center;gap:6px'>
    <span style='width:14px;height:14px;border-radius:50%;background:#eaf3fb;border:2px solid {WD_BLUE};display:inline-block'></span>
    <span style='color:{WD_BLUE}'>Terminal entity</span>
      </span>
      <span style='display:flex;align-items:center;gap:6px'>
    <span style='width:14px;height:14px;border-radius:2px;background:#edf7f1;border:2px solid {WD_GREEN};display:inline-block'></span>
    <span style='color:{WD_GREEN}'>Connector node</span>
      </span>
      <span style='display:flex;align-items:center;gap:6px'>
    <span style='display:inline-flex;align-items:center;gap:2px'>
      <span style='width:22px;height:2px;background:{WD_RED};display:inline-block'></span>
      <span style='width:0;height:0;border-top:4px solid transparent;border-bottom:4px solid transparent;border-left:7px solid {WD_RED};display:inline-block'></span>
    </span>
    <span style='color:{WD_RED}'>Property (click for details)</span>
      </span>
    </div>
    """)
    return


@app.cell
def vis_graph(G, graph_result, terminals):
    mo.stop(graph_result is None)

    with mo.status.spinner(title="Fetching labels…"):
        html = build_graph_html(G, terminals)

    mo.Html(html)
    return


@app.cell
def sparql_output(G, graph_result):
    mo.stop(graph_result is None)

    sparql_query = build_sparql(G)

    mo.md(f"""
    ## SPARQL to reproduce on QLever

    ```sparql
    {sparql_query}
    ```

    [▶ Open in QLever](https://qlever.cs.uni-freiburg.de/wikidata?query={requests.utils.quote(sparql_query)})
    """)
    return


@app.cell
def footer():
    mo.md("""
    ---
    **Data:**
    <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a> |
    **Code:**
    <a href="https://github.com/Adafede/marimo/blob/main/apps/wikidata_connector.py" style="color:#339966;">wikidata_connector.py</a> |
    **External tools:**
    <a href="https://qlever.dev/wikidata" style="color:#006699;">QLever</a> |
    **License:**
    <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#484848;">CC0 1.0</a> for data &
    <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#484848;">AGPL-3.0</a> for code
    """)
    return


if __name__ == "__main__":
    app.run()
