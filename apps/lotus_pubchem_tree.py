# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "polars==1.37.1",
# ]
# [tool.marimo.runtime]
# output_max_bytes = 300_000_000
# ///

"""
LOTUS PubChem Tree Generator

Generates JSON files for PubChem classification matching:
https://pubchem.ncbi.nlm.nih.gov/classification/#hid=115

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

Usage:
    # Remote CLI export (auto-installs deps, fetches, builds, saves)
    uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v

    # Remote GUI (interactive)
    uvx marimo run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py

    # Local GUI
    marimo run lotus_pubchem_tree.py

    # Local CLI export
    python lotus_pubchem_tree.py export -o ./output -v
"""

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full", app_title="LOTUS PubChem Tree Generator")

with app.setup:
    import marimo as mo
    import polars as pl
    import io
    import json
    import time
    import sys
    from dataclasses import dataclass
    from datetime import datetime

    _USE_LOCAL = True
    if _USE_LOCAL:
        sys.path.insert(0, ".")

    from modules.net.sparql.execute_with_retry import execute_with_retry
    from modules.knowledge.wikidata.url.constants import (
        ENTITY_PREFIX as WIKIDATA_ENTITY_PREFIX,
    )

    IS_PYODIDE = "pyodide" in sys.modules
    if IS_PYODIDE:
        import pyodide_http

        pyodide_http.patch_all()

    CONFIG = {
        "app_version": "0.1.0",
        "app_name": "LOTUS PubChem Tree Generator",
        "qlever_endpoint": "https://qlever.dev/api/wikidata",
        "preview_max_root_nodes": 30,
        "preview_max_children": 20,
        "preview_max_depth": 3,
        # Pyodide memory limits - significantly reduce data for browser mode
        "pyodide_max_compound_taxon_pairs": 10000,
    }

    # ========================================================================
    # SPARQL QUERIES
    # ========================================================================

    QUERY_COMPOUND_INCHIKEY_TAXON = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?compound ?compound_inchikey ?taxon WHERE {
      ?compound wdt:P235 ?compound_inchikey ;
            wdt:P703 ?taxon .
    }
    """

    QUERY_TAXON_NCBI = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?taxon ?taxon_ncbi WHERE {
      ?taxon wdt:P685 ?taxon_ncbi .
    }
    """

    QUERY_TAXON_PARENT = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?taxon ?taxon_parent WHERE {
      ?taxon wdt:P171 ?taxon_parent .
      ?taxon_parent wdt:P171* wd:Q2382443 .
    }
    """

    QUERY_TAXON_NAME = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?taxon ?taxon_name WHERE {
      ?taxon wdt:P225 ?taxon_name .
    }
    """

    QUERY_COMPOUND_PARENT = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?compound ?compound_parent WHERE {
      ?compound wdt:P279 ?compound_parent .
      ?compound_parent wdt:P279* wd:Q11173 .
    }
    """

    QUERY_COMPOUND_LABEL = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?compound ?compound_label WHERE {
      ?compound wdt:P279* wd:Q11173 .
      ?compound rdfs:label ?compound_label .
      FILTER (LANG(?compound_label) IN ("en", "mul"))
    }
    """

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    QUERY_SCHEMAS = {
        "compound_taxon": {
            "compound": pl.Utf8,
            "compound_inchikey": pl.Utf8,
            "taxon": pl.Utf8,
        },
        "taxon_ncbi": {
            "taxon": pl.Utf8,
            "taxon_ncbi": pl.Utf8,
        },
        "taxon_parent": {
            "taxon": pl.Utf8,
            "taxon_parent": pl.Utf8,
        },
        "taxon_name": {
            "taxon": pl.Utf8,
            "taxon_name": pl.Utf8,
        },
        "compound_parent": {
            "compound": pl.Utf8,
            "compound_parent": pl.Utf8,
        },
        "compound_label": {
            "compound": pl.Utf8,
            "compound_label": pl.Utf8,
        },
    }

    def extract_qid(url: str) -> str:
        """Extract QID from Wikidata URL."""
        if url and url.startswith(WIKIDATA_ENTITY_PREFIX):
            return url.replace(WIKIDATA_ENTITY_PREFIX, "")
        return url

    def execute_query(
        query: str,
        endpoint: str,
        schema_name: str | None = None,
    ) -> pl.LazyFrame:
        """Execute SPARQL query and return LazyFrame."""
        csv_bytes = execute_with_retry(query, endpoint, timeout=600)
        if not csv_bytes or len(csv_bytes) < 10:
            return pl.LazyFrame()

        schema = QUERY_SCHEMAS.get(schema_name) if schema_name else None
        return pl.scan_csv(
            io.BytesIO(csv_bytes),
            rechunk=False,
            schema_overrides=schema,
            infer_schema_length=10000,
        )

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    @dataclass
    class LOTUSData:
        """Container for all fetched LOTUS data."""

        compound_taxon: pl.LazyFrame
        taxon_ncbi: pl.LazyFrame
        taxon_parent: pl.LazyFrame
        taxon_name: pl.LazyFrame
        compound_parent: pl.LazyFrame
        compound_label: pl.LazyFrame

    def fetch_all_data(endpoint: str, progress_callback=None) -> LOTUSData:
        """Fetch all required data from Wikidata."""
        queries = [
            (
                "compound_taxon",
                QUERY_COMPOUND_INCHIKEY_TAXON,
                "Fetching compound-InChIKey-taxon triplets...",
            ),
            ("taxon_ncbi", QUERY_TAXON_NCBI, "Fetching taxon-NCBI pairs..."),
            (
                "taxon_parent",
                QUERY_TAXON_PARENT,
                "Fetching taxon-parent pairs (under Biota)...",
            ),
            ("taxon_name", QUERY_TAXON_NAME, "Fetching taxon-name pairs..."),
            (
                "compound_parent",
                QUERY_COMPOUND_PARENT,
                "Fetching compound-parent pairs (under chemical compound)...",
            ),
            (
                "compound_label",
                QUERY_COMPOUND_LABEL,
                "Fetching compound-label pairs...",
            ),
        ]

        results = {}
        for name, query, msg in queries:
            if progress_callback:
                progress_callback(msg)
            results[name] = execute_query(query, endpoint, schema_name=name)

        return LOTUSData(**results)

    # ========================================================================
    # DATA PROCESSING
    # ========================================================================

    def extract_qids_from_lazyframe(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
        """Extract QIDs from Wikidata URLs in a column."""
        return lf.with_columns(
            pl.col(col)
            .str.replace(WIKIDATA_ENTITY_PREFIX, "", literal=True)
            .alias(col),
        )

    def process_compound_taxon(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process compound-taxon triplets."""
        return lf.pipe(extract_qids_from_lazyframe, "compound").pipe(
            extract_qids_from_lazyframe,
            "taxon",
        )

    def process_taxon_ncbi(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process taxon-NCBI pairs."""
        return lf.pipe(extract_qids_from_lazyframe, "taxon")

    def process_taxon_parent(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process taxon-parent pairs."""
        return lf.pipe(extract_qids_from_lazyframe, "taxon").pipe(
            extract_qids_from_lazyframe,
            "taxon_parent",
        )

    def process_taxon_name(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process taxon-name pairs."""
        return lf.pipe(extract_qids_from_lazyframe, "taxon")

    def process_compound_parent(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process compound-parent pairs."""
        return lf.pipe(extract_qids_from_lazyframe, "compound").pipe(
            extract_qids_from_lazyframe,
            "compound_parent",
        )

    def process_compound_label(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process compound-label pairs."""
        return lf.pipe(extract_qids_from_lazyframe, "compound")

    # ========================================================================
    # TREE BUILDING
    # ========================================================================

    def build_biological_tree(
        compound_taxon: pl.DataFrame,
        taxon_ncbi: pl.DataFrame,
        taxon_parent: pl.DataFrame,
        taxon_name: pl.DataFrame,
    ) -> list[dict]:
        """
        Build biological tree JSON for PubChem.

        Only includes nodes that have InChIKeys directly or in their descendants.
        All taxa are constrained to be under Biota (Q2382443).

        Returns a hierarchical tree structure where each node contains:
        - NCBI_TaxID: NCBI taxonomy ID
        - Wikidata_QID: Wikidata QID
        - Name: Taxonomic name
        - InChIKeys: List of compound InChIKeys found in this taxon
        - Children: List of child nodes
        """
        # Build parent-child relationships
        parent_map = (
            taxon_parent.group_by("taxon_parent")
            .agg(
                pl.col("taxon").alias("children"),
            )
            .to_dict(as_series=False)
        )
        parent_dict = dict(zip(parent_map["taxon_parent"], parent_map["children"]))

        # Build taxon name mapping
        name_map = dict(
            zip(
                taxon_name["taxon"].to_list(),
                taxon_name["taxon_name"].to_list(),
            ),
        )

        # Build taxon NCBI mapping
        ncbi_map = dict(
            zip(
                taxon_ncbi["taxon"].to_list(),
                taxon_ncbi["taxon_ncbi"].to_list(),
            ),
        )

        # Build taxon to InChIKeys mapping
        compound_by_taxon = (
            compound_taxon.group_by("taxon")
            .agg(
                pl.col("compound_inchikey").alias("inchikeys"),
            )
            .to_dict(as_series=False)
        )
        inchikey_map = dict(
            zip(
                compound_by_taxon["taxon"],
                compound_by_taxon["inchikeys"],
            ),
        )

        # Find root taxa (those without parents in our set)
        all_taxa = set(taxon_parent["taxon"].to_list())
        all_parents = set(taxon_parent["taxon_parent"].to_list())
        roots = all_parents - all_taxa

        def build_node(taxon_qid: str, visited: set) -> dict | None:
            """Build node, returning None if no InChIKeys in subtree."""
            if taxon_qid in visited:
                return None
            visited.add(taxon_qid)

            inchikeys = inchikey_map.get(taxon_qid, [])
            has_own_inchikeys = len(inchikeys) > 0

            # Build children first to check if any have InChIKeys
            children = parent_dict.get(taxon_qid, [])
            child_nodes = []
            for child in children:
                child_node = build_node(child, visited)
                if child_node:
                    child_nodes.append(child_node)

            # Only include this node if it has InChIKeys or has valid children
            if not has_own_inchikeys and not child_nodes:
                return None

            ncbi_id = ncbi_map.get(taxon_qid)
            name = name_map.get(taxon_qid, taxon_qid)

            node = {
                "NCBI_TaxID": ncbi_id,
                "Wikidata_QID": taxon_qid,
                "Name": name,
            }

            if inchikeys:
                node["InChIKeys"] = inchikeys

            if child_nodes:
                node["Children"] = sorted(child_nodes, key=lambda x: x.get("Name", ""))

            return node

        # Build tree from roots
        tree = []
        visited = set()
        for root in sorted(roots):
            node = build_node(root, visited)
            if node:
                tree.append(node)

        return sorted(tree, key=lambda x: x.get("Name", ""))

    def build_compound_tree(
        compound_taxon: pl.DataFrame,
        compound_parent: pl.DataFrame,
        compound_label: pl.DataFrame,
    ) -> list[dict]:
        """
        Build chemical compound tree JSON for PubChem.

        Only includes nodes that have InChIKeys directly or in their descendants.
        All compounds are constrained to be under chemical compound (Q11173).

        Returns a hierarchical tree structure where each node contains:
        - Wikidata_QID: Wikidata QID
        - Label: Compound label/name
        - InChIKeys: List of InChIKeys for compounds in this class
        - Children: List of child compound classes
        """
        # Build parent-child relationships
        parent_map = (
            compound_parent.group_by("compound_parent")
            .agg(
                pl.col("compound").alias("children"),
            )
            .to_dict(as_series=False)
        )
        parent_dict = dict(zip(parent_map["compound_parent"], parent_map["children"]))

        # Build compound label mapping
        label_map = dict(
            zip(
                compound_label["compound"].to_list(),
                compound_label["compound_label"].to_list(),
            ),
        )

        # Build compound to InChIKeys mapping (from compound_taxon)
        compound_by_class = (
            compound_taxon.group_by("compound")
            .agg(
                pl.col("compound_inchikey").unique().alias("inchikeys"),
            )
            .to_dict(as_series=False)
        )
        inchikey_map = dict(
            zip(
                compound_by_class["compound"],
                compound_by_class["inchikeys"],
            ),
        )

        # Find root compounds
        all_compounds = set(compound_parent["compound"].to_list())
        all_parents = set(compound_parent["compound_parent"].to_list())
        roots = all_parents - all_compounds

        def build_node(compound_qid: str, visited: set) -> dict | None:
            """Build node, returning None if no InChIKeys in subtree."""
            if compound_qid in visited:
                return None
            visited.add(compound_qid)

            inchikeys = inchikey_map.get(compound_qid, [])
            has_own_inchikeys = len(inchikeys) > 0

            # Build children first to check if any have InChIKeys
            children = parent_dict.get(compound_qid, [])
            child_nodes = []
            for child in children:
                child_node = build_node(child, visited)
                if child_node:
                    child_nodes.append(child_node)

            # Only include this node if it has InChIKeys or has valid children
            if not has_own_inchikeys and not child_nodes:
                return None

            label = label_map.get(compound_qid, compound_qid)

            node = {
                "Wikidata_QID": compound_qid,
                "Label": label,
            }

            if inchikeys:
                node["InChIKeys"] = inchikeys

            if child_nodes:
                node["Children"] = sorted(child_nodes, key=lambda x: x.get("Label", ""))

            return node

        # Build tree from roots
        tree = []
        visited = set()
        for root in sorted(roots):
            node = build_node(root, visited)
            if node:
                tree.append(node)

        return sorted(tree, key=lambda x: x.get("Label", ""))

    def tree_to_display(tree: list[dict], label_key: str) -> tuple[dict, int, int]:
        """
        Convert tree to mo.tree() compatible format for display.

        Returns (display_dict, shown_nodes, total_nodes).
        """
        max_depth = CONFIG["preview_max_depth"]
        max_children = CONFIG["preview_max_children"]
        max_root = CONFIG["preview_max_root_nodes"]

        shown_count = 0

        def convert_node(node: dict, depth: int) -> dict:
            nonlocal shown_count
            shown_count += 1

            label = node.get(label_key, node.get("Wikidata_QID", "Unknown"))
            n_inchikeys = len(node.get("InChIKeys", []))

            display_label = f"{label}"
            if n_inchikeys > 0:
                display_label += f" ({n_inchikeys} InChIKeys)"

            result = {}
            if "Children" in node and depth < max_depth:
                children = node["Children"]
                for child in children[:max_children]:
                    child_dict = convert_node(child, depth + 1)
                    result.update(child_dict)
                if len(children) > max_children:
                    result[f"⋯ {len(children) - max_children} more children"] = {}

            return {display_label: result}

        display_dict = {}
        for node in tree[:max_root]:
            display_dict.update(convert_node(node, 0))
        if len(tree) > max_root:
            display_dict[f"⋯ {len(tree) - max_root} more root nodes"] = {}

        return display_dict, shown_count, count_tree_nodes(tree)

    # ========================================================================
    # STATISTICS
    # ========================================================================

    @dataclass
    class DataStats:
        """Statistics about the fetched data."""

        n_compounds: int = 0
        n_taxa: int = 0
        n_compound_taxon_pairs: int = 0
        n_taxa_with_ncbi: int = 0
        n_taxon_parent_pairs: int = 0
        n_taxa_with_names: int = 0
        n_compound_parent_pairs: int = 0
        n_compounds_with_labels: int = 0

    def compute_stats(data: LOTUSData) -> DataStats:
        """Compute statistics from the data."""
        compound_taxon_df = data.compound_taxon.collect()
        taxon_ncbi_df = data.taxon_ncbi.collect()
        taxon_parent_df = data.taxon_parent.collect()
        taxon_name_df = data.taxon_name.collect()
        compound_parent_df = data.compound_parent.collect()
        compound_label_df = data.compound_label.collect()

        return DataStats(
            n_compounds=compound_taxon_df.select(pl.col("compound").n_unique()).item(),
            n_taxa=compound_taxon_df.select(pl.col("taxon").n_unique()).item(),
            n_compound_taxon_pairs=len(compound_taxon_df),
            n_taxa_with_ncbi=len(taxon_ncbi_df),
            n_taxon_parent_pairs=len(taxon_parent_df),
            n_taxa_with_names=len(taxon_name_df),
            n_compound_parent_pairs=len(compound_parent_df),
            n_compounds_with_labels=len(compound_label_df),
        )

    def count_tree_nodes(tree: list[dict]) -> int:
        """Count total nodes in the tree."""
        count = 0
        for node in tree:
            count += 1
            if "Children" in node:
                count += count_tree_nodes(node["Children"])
        return count


@app.cell
def md_title():
    _pyodide_warning = ""
    if IS_PYODIDE:
        _pyodide_warning = f"""
> ⚠️ **Browser Mode**: Limited to {CONFIG["pyodide_max_compound_taxon_pairs"]:,} compound-taxon pairs for preview.
> For full trees, use: `uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v`
"""

    mo.md(f"""
    # LOTUS PubChem Tree Generator

    This app generates hierarchical JSON files for PubChem classification matching
    [PubChem Classification](https://pubchem.ncbi.nlm.nih.gov/classification/#hid=115).

    It produces two JSON files:
    1. **Biological Tree**: Hierarchical taxonomy (under [Biota](https://www.wikidata.org/wiki/Q2382443)) with associated compound InChIKeys
    2. **Chemical Tree**: Hierarchical compound classes (under [chemical compound](https://www.wikidata.org/wiki/Q11173)) with associated InChIKeys

    *Only nodes with InChIKeys (directly or in descendants) are included.*
    {_pyodide_warning}
    """)
    return


@app.cell
def ui_controls():
    run_button = mo.ui.run_button(label="Fetch Data from Wikidata")
    run_button
    return (run_button,)


@app.cell
def fetch_data(run_button):
    mo.stop(not run_button.value, mo.md("Click **Fetch Data from Wikidata** to start"))

    start_time = time.time()

    with mo.status.spinner("Fetching data from Wikidata...") as _spinner:

        def progress_callback(msg):
            _spinner.update(msg)

        data = fetch_all_data(CONFIG["qlever_endpoint"], progress_callback)

    with mo.status.spinner("Processing data..."):
        data = LOTUSData(
            compound_taxon=process_compound_taxon(data.compound_taxon),
            taxon_ncbi=process_taxon_ncbi(data.taxon_ncbi),
            taxon_parent=process_taxon_parent(data.taxon_parent),
            taxon_name=process_taxon_name(data.taxon_name),
            compound_parent=process_compound_parent(data.compound_parent),
            compound_label=process_compound_label(data.compound_label),
        )

    with mo.status.spinner("Computing statistics..."):
        stats = compute_stats(data)

    elapsed = round(time.time() - start_time, 2)
    mo.md(f"Data fetched and processed in **{elapsed}s**")
    return data, stats


@app.cell
def display_stats(stats):
    mo.stop(stats is None)

    mo.vstack(
        [
            mo.md("## Data Statistics"),
            mo.hstack(
                [
                    mo.stat(value=f"{stats.n_compounds:,}", label="Unique Compounds"),
                    mo.stat(value=f"{stats.n_taxa:,}", label="Unique Taxa"),
                    mo.stat(
                        value=f"{stats.n_compound_taxon_pairs:,}",
                        label="Compound-Taxon Pairs",
                    ),
                ],
                gap=0,
                wrap=True,
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{stats.n_taxa_with_ncbi:,}",
                        label="Taxa with NCBI ID",
                    ),
                    mo.stat(
                        value=f"{stats.n_taxon_parent_pairs:,}",
                        label="Taxon-Parent Pairs",
                    ),
                    mo.stat(
                        value=f"{stats.n_taxa_with_names:,}",
                        label="Taxa with Names",
                    ),
                ],
                gap=0,
                wrap=True,
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{stats.n_compound_parent_pairs:,}",
                        label="Compound-Parent Pairs",
                    ),
                    mo.stat(
                        value=f"{stats.n_compounds_with_labels:,}",
                        label="Compounds with Labels",
                    ),
                ],
                gap=0,
                wrap=True,
            ),
        ],
    )
    return


@app.cell
def build_trees_button(data):
    mo.stop(data is None)

    if IS_PYODIDE:
        build_trees_btn = mo.ui.run_button(label="Build Trees (Limited Preview)")
    else:
        build_trees_btn = mo.ui.run_button(label="Build Trees")
    build_trees_btn
    return (build_trees_btn,)


@app.cell
def build_trees(build_trees_btn, data):
    mo.stop(data is None)
    mo.stop(
        build_trees_btn is None or not build_trees_btn.value,
        mo.md("Click **Build Trees** to generate the tree structures"),
    )

    # In Pyodide, sample data to reduce memory usage
    if IS_PYODIDE:
        max_pairs = CONFIG["pyodide_max_compound_taxon_pairs"]
        compound_taxon_df = data.compound_taxon.head(max_pairs).collect()
        _pyodide_note = f" (limited to {max_pairs:,} compound-taxon pairs)"
    else:
        compound_taxon_df = data.compound_taxon.collect()
        _pyodide_note = ""

    with mo.status.spinner("Building biological tree..."):
        taxon_ncbi_df = data.taxon_ncbi.collect()
        taxon_parent_df = data.taxon_parent.collect()
        taxon_name_df = data.taxon_name.collect()

        biological_tree = build_biological_tree(
            compound_taxon_df,
            taxon_ncbi_df,
            taxon_parent_df,
            taxon_name_df,
        )

    with mo.status.spinner("Building chemical tree..."):
        compound_parent_df = data.compound_parent.collect()
        compound_label_df = data.compound_label.collect()

        chemical_tree = build_compound_tree(
            compound_taxon_df,
            compound_parent_df,
            compound_label_df,
        )

    bio_nodes = count_tree_nodes(biological_tree)
    chem_nodes = count_tree_nodes(chemical_tree)

    _output = [
        mo.md(f"""
## Trees Built{_pyodide_note}

- **Biological Tree**: {len(biological_tree)} root nodes, {bio_nodes:,} total nodes
- **Chemical Tree**: {len(chemical_tree)} root nodes, {chem_nodes:,} total nodes
        """),
    ]

    if IS_PYODIDE:
        _output.append(
            mo.callout(
                mo.md("""
**This is a limited preview.** For full trees, use the CLI:
```bash
uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v
```
                """),
                kind="info",
            ),
        )

    mo.vstack(_output)
    return biological_tree, chemical_tree


@app.cell
def display_previews(biological_tree, chemical_tree):
    mo.stop(biological_tree is None or chemical_tree is None)

    bio_display, bio_shown, bio_total = tree_to_display(biological_tree, "Name")
    chem_display, chem_shown, chem_total = tree_to_display(chemical_tree, "Label")

    mo.vstack(
        [
            mo.callout(
                mo.md(f"""
            **Preview is truncated for performance.** Showing ~{bio_shown + chem_shown:,} nodes out of {bio_total + chem_total:,} total.
            Download the JSON files for the complete trees.
            """),
                kind="info",
            ),
            mo.ui.tabs(
                {
                    f"Biological Tree ({bio_total:,} nodes)": mo.tree(bio_display),
                    f"Chemical Tree ({chem_total:,} nodes)": mo.tree(chem_display),
                },
            ),
        ],
    )
    return


@app.cell
def download_buttons(biological_tree, chemical_tree):
    mo.stop(biological_tree is None or chemical_tree is None)

    date_str = datetime.now().strftime("%Y%m%d")

    biological_json = json.dumps(
        {
            "metadata": {
                "name": "LOTUS Biological Tree",
                "description": "Hierarchical taxonomy (under Biota Q2382443) with associated compound InChIKeys",
                "source": "Wikidata/LOTUS",
                "generated": datetime.now().isoformat(),
                "version": CONFIG["app_version"],
                "root_constraint": "wdt:P171* wd:Q2382443",
            },
            "tree": biological_tree,
        },
        indent=2,
    )

    chemical_json = json.dumps(
        {
            "metadata": {
                "name": "LOTUS Chemical Tree",
                "description": "Hierarchical compound classification (under chemical compound Q11173) with associated InChIKeys",
                "source": "Wikidata/LOTUS",
                "generated": datetime.now().isoformat(),
                "version": CONFIG["app_version"],
                "root_constraint": "wdt:P279* wd:Q11173",
            },
            "tree": chemical_tree,
        },
        indent=2,
    )

    mo.vstack(
        [
            mo.md("## Download Trees"),
            mo.hstack(
                [
                    mo.download(
                        label="Biological Tree JSON",
                        filename=f"{date_str}_lotus_biological_tree.json",
                        mimetype="application/json",
                        data=lambda: biological_json.encode("utf-8"),
                    ),
                    mo.download(
                        label="Chemical Tree JSON",
                        filename=f"{date_str}_lotus_chemical_tree.json",
                        mimetype="application/json",
                        data=lambda: chemical_json.encode("utf-8"),
                    ),
                ],
                gap=2,
            ),
        ],
    )
    return


@app.cell
def cli_usage():
    mo.accordion(
        {
            "Programmatic / CLI Usage": mo.md("""
**Remote CLI export (one-liner, auto-installs deps):**
```bash
uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v
```

**Remote GUI (interactive, limited preview in browser):**
```bash
uvx marimo run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py
```

---

**Local usage:**
```bash
# GUI
marimo run lotus_pubchem_tree.py

# CLI export
python lotus_pubchem_tree.py export -o ./output -v
```
        """),
        },
    )
    return


@app.cell
def footer():
    mo.md("""
    ---
    **Data:**
    <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> &
    <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a> |
    **Code:**
    <a href="https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py" style="color:#339966;">lotus_pubchem_tree.py</a> |
    **External tools:**
    <a href="https://qlever.dev/wikidata" style="color:#006699;">QLever</a> |
    **License:**
    <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#484848;">CC0 1.0</a> for data &
    <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#484848;">AGPL-3.0</a> for code
    """)
    return


def main():
    """Entry point for CLI mode."""
    import argparse
    from pathlib import Path

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        parser = argparse.ArgumentParser(
            description="Export LOTUS PubChem trees via CLI",
            epilog="""
Examples:
  uv run lotus_pubchem_tree.py export -o ./output -v
  python lotus_pubchem_tree.py export -o ./output -v
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("export", help="Export command")
        parser.add_argument("--output-dir", "-o", help="Output directory", default=".")
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Verbose output"
        )
        args = parser.parse_args()

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if args.verbose:
                print("=" * 60, file=sys.stderr)
                print("LOTUS PubChem Tree Generator - CLI Export", file=sys.stderr)
                print("=" * 60, file=sys.stderr)
                print(f"Output directory: {output_dir.absolute()}", file=sys.stderr)
                print(f"Endpoint: {CONFIG['qlever_endpoint']}", file=sys.stderr)
                print("", file=sys.stderr)

            def progress_callback(msg):
                if args.verbose:
                    print(f"  → {msg}", file=sys.stderr)

            if args.verbose:
                print("Fetching data from Wikidata...", file=sys.stderr)

            data = fetch_all_data(CONFIG["qlever_endpoint"], progress_callback)

            if args.verbose:
                print("\nProcessing data...", file=sys.stderr)

            data = LOTUSData(
                compound_taxon=process_compound_taxon(data.compound_taxon),
                taxon_ncbi=process_taxon_ncbi(data.taxon_ncbi),
                taxon_parent=process_taxon_parent(data.taxon_parent),
                taxon_name=process_taxon_name(data.taxon_name),
                compound_parent=process_compound_parent(data.compound_parent),
                compound_label=process_compound_label(data.compound_label),
            )

            if args.verbose:
                stats = compute_stats(data)
                print(f"  Compounds: {stats.n_compounds:,}", file=sys.stderr)
                print(f"  Taxa: {stats.n_taxa:,}", file=sys.stderr)
                print(
                    f"  Compound-Taxon pairs: {stats.n_compound_taxon_pairs:,}",
                    file=sys.stderr,
                )

            if args.verbose:
                print("\nBuilding biological tree...", file=sys.stderr)

            compound_taxon_df = data.compound_taxon.collect()
            taxon_ncbi_df = data.taxon_ncbi.collect()
            taxon_parent_df = data.taxon_parent.collect()
            taxon_name_df = data.taxon_name.collect()

            biological_tree = build_biological_tree(
                compound_taxon_df,
                taxon_ncbi_df,
                taxon_parent_df,
                taxon_name_df,
            )

            if args.verbose:
                bio_nodes = count_tree_nodes(biological_tree)
                print(f"  Root nodes: {len(biological_tree)}", file=sys.stderr)
                print(f"  Total nodes: {bio_nodes:,}", file=sys.stderr)

            if args.verbose:
                print("\nBuilding chemical tree...", file=sys.stderr)

            compound_parent_df = data.compound_parent.collect()
            compound_label_df = data.compound_label.collect()

            chemical_tree = build_compound_tree(
                compound_taxon_df,
                compound_parent_df,
                compound_label_df,
            )

            if args.verbose:
                chem_nodes = count_tree_nodes(chemical_tree)
                print(f"  Root nodes: {len(chemical_tree)}", file=sys.stderr)
                print(f"  Total nodes: {chem_nodes:,}", file=sys.stderr)

            date_str = datetime.now().strftime("%Y%m%d")

            biological_output = {
                "metadata": {
                    "name": "LOTUS Biological Tree",
                    "description": "Hierarchical taxonomy (under Biota Q2382443) with associated compound InChIKeys",
                    "source": "Wikidata/LOTUS",
                    "generated": datetime.now().isoformat(),
                    "version": CONFIG["app_version"],
                    "root_constraint": "wdt:P171* wd:Q2382443",
                },
                "tree": biological_tree,
            }

            chemical_output = {
                "metadata": {
                    "name": "LOTUS Chemical Tree",
                    "description": "Hierarchical compound classification (under chemical compound Q11173) with associated InChIKeys",
                    "source": "Wikidata/LOTUS",
                    "generated": datetime.now().isoformat(),
                    "version": CONFIG["app_version"],
                    "root_constraint": "wdt:P279* wd:Q11173",
                },
                "tree": chemical_tree,
            }

            biological_path = output_dir / f"{date_str}_lotus_biological_tree.json"
            chemical_path = output_dir / f"{date_str}_lotus_chemical_tree.json"

            if args.verbose:
                print("\nWriting output files...", file=sys.stderr)

            biological_path.write_text(json.dumps(biological_output, indent=2))
            chemical_path.write_text(json.dumps(chemical_output, indent=2))

            if args.verbose:
                print(f"  ✓ {biological_path}", file=sys.stderr)
                print(f"  ✓ {chemical_path}", file=sys.stderr)
                print("\nDone!", file=sys.stderr)
            else:
                print(biological_path)
                print(chemical_path)

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)
    else:
        app.run()


if __name__ == "__main__":
    main()
