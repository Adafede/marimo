# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "polars==1.37.1",
# ]
# [tool.marimo.runtime]
# output_max_bytes = 1_073_741_824
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

    # Metadata for provenance and reproducibility
    METADATA = {
        "project": "LOTUS",
        "project_url": "https://lotus.nprod.net/",
        "wikidata_item": "Q104225190",
        "source": "Wikidata via QLever SPARQL endpoint",
        "license_data": "CC0 1.0 Universal",
        "license_code": "AGPL-3.0",
        "pubchem_target": "https://pubchem.ncbi.nlm.nih.gov/classification/#hid=115",
        "constraints": {
            "biological_tree": "Taxa under Biota (Q2382443) with P171 (parent taxon)",
            "chemical_tree": "Compounds under chemical compound (Q11173) with P279 (subclass of)",
        },
        "wikidata_properties": {
            "P171": "parent taxon",
            "P225": "taxon name",
            "P233": "canonical SMILES",
            "P235": "InChIKey",
            "P279": "subclass of",
            "P685": "NCBI taxonomy ID",
            "P703": "found in taxon",
            "P2017": "isomeric SMILES",
            "P8533": "SMARTS",
            "P10718": "CXSMILES",
        },
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

    QUERY_COMPOUND_SMILES_CAN = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?compound ?compound_smiles_can WHERE {
          ?compound wdt:P233 ?compound_smiles_can .
        }
    """

    QUERY_COMPOUND_SMILES_ISO = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?compound ?compound_smiles_iso WHERE {
          ?compound wdt:P2017 ?compound_smiles_iso .
        }
    """

    QUERY_COMPOUND_SMARTS = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?compound ?compound_smarts WHERE {
          ?compound wdt:P8533 ?compound_smarts .
        }
    """

    QUERY_COMPOUND_CXSMILES = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?compound ?compound_cxsmiles WHERE {
          ?compound wdt:P10718 ?compound_cxsmiles .
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
    SELECT DISTINCT ?compound ?compound_label ?lang WHERE {
      ?compound wdt:P279* wd:Q11173 .
      ?compound rdfs:label ?compound_label .
      BIND(LANG(?compound_label) AS ?lang)
      FILTER (?lang IN ("en", "mul"))
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
        "compound_smiles_can": {
            "compound": pl.Utf8,
            "compound_smiles_can": pl.Utf8,
        },
        "compound_smiles_iso": {
            "compound": pl.Utf8,
            "compound_smiles_iso": pl.Utf8,
        },
        "compound_smarts": {
            "compound": pl.Utf8,
            "compound_smarts": pl.Utf8,
        },
        "compound_cxsmiles": {
            "compound": pl.Utf8,
            "compound_cxsmiles": pl.Utf8,
        },
        "compound_parent": {
            "compound": pl.Utf8,
            "compound_parent": pl.Utf8,
        },
        "compound_label": {
            "compound": pl.Utf8,
            "compound_label": pl.Utf8,
            "lang": pl.Utf8,
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
    # DATA STRUCTURES
    # ========================================================================
    # Naming conventions:
    #   - *_map: dict mapping QID to value(s), e.g., inchikey_map, smiles_map
    #   - Descriptors fields: singular (type name), hold list[str] | None
    #   - Local variables: plural when holding lists, e.g., inchikeys, children
    # ========================================================================

    @dataclass
    class LOTUSData:
        """Container for all fetched LOTUS data as LazyFrames."""

        compound_taxon: pl.LazyFrame
        taxon_ncbi: pl.LazyFrame
        taxon_parent: pl.LazyFrame
        taxon_name: pl.LazyFrame
        compound_smiles_can: pl.LazyFrame
        compound_smiles_iso: pl.LazyFrame
        compound_smarts: pl.LazyFrame
        compound_cxsmiles: pl.LazyFrame
        compound_parent: pl.LazyFrame
        compound_label: pl.LazyFrame

    @dataclass
    class Identifiers:
        """Identifiers for an entity (compound or taxon)."""

        wikidata_qid: str
        ncbi_taxid: str | None = None

        def to_dict(self) -> dict:
            """Convert to dictionary, omitting None fields."""
            result = {"Wikidata_QID": self.wikidata_qid}
            if self.ncbi_taxid:
                result["NCBI_TaxID"] = self.ncbi_taxid
            return result

    @dataclass
    class Descriptors:
        """
        Chemical structure descriptors for a compound.

        Field names are singular (descriptor type), but hold lists since a compound
        may have multiple values (rare). Output uses single value when unique.
        """

        inchikey: list[str] | None = None
        smiles: list[str] | None = None
        smarts: list[str] | None = None
        cxsmiles: list[str] | None = None

        @staticmethod
        def _simplify(values: list[str] | None) -> str | list[str] | None:
            """Return single value if list has one item, else return list."""
            if not values:
                return None
            if len(values) == 1:
                return values[0]
            return values

        def to_dict(self) -> dict:
            """Convert to dictionary. Uses single value when unique, list when multiple."""
            result = {}
            if self.inchikey:
                result["InChIKey"] = self._simplify(self.inchikey)
            if self.smiles:
                result["SMILES"] = self._simplify(self.smiles)
            if self.smarts:
                result["SMARTS"] = self._simplify(self.smarts)
            if self.cxsmiles:
                result["CXSMILES"] = self._simplify(self.cxsmiles)
            return result

        def is_empty(self) -> bool:
            """Check if all descriptors are empty."""
            return not (self.inchikey or self.smiles or self.smarts or self.cxsmiles)

    @dataclass
    class Taxon:
        """A biological taxon with identifiers and name."""

        identifiers: Identifiers
        name: str

        @property
        def qid(self) -> str:
            """Shortcut to get the QID."""
            return self.identifiers.wikidata_qid

        def to_node_dict(
            self,
            compounds: list[dict] = None,
            children: list[dict] = None,
        ) -> dict:
            """Convert to tree node dictionary. Name first for readability."""
            node = {
                "Name": self.name,
                "Identifiers": self.identifiers.to_dict(),
            }
            if compounds:
                node["Compounds"] = compounds
            if children:
                node["Children"] = children
            return node

    def build_compounds_with_taxa(
        compound_taxon: pl.DataFrame,
    ) -> tuple[set[str], dict[str, list[str]]]:
        """
        Identify compounds that have InChIKeys AND are associated with taxa.

        These are the "leaf" compounds that must appear in the trees.
        Returns:
            - Set of compound QIDs with valid InChIKeys+taxa
            - Dict mapping compound QID to list of InChIKeys
        """
        # Group InChIKeys by compound
        compound_inchikeys = (
            compound_taxon.group_by("compound")
            .agg(pl.col("compound_inchikey").unique().alias("inchikeys"))
            .to_dict(as_series=False)
        )

        inchikey_map: dict[str, list[str]] = {}
        compounds_with_taxa: set[str] = set()

        for qid, inchikeys in zip(
            compound_inchikeys["compound"],
            compound_inchikeys["inchikeys"],
        ):
            # Filter out None/empty InChIKeys
            valid_inchikeys = [ik for ik in inchikeys if ik]
            if valid_inchikeys:
                inchikey_map[qid] = valid_inchikeys
                compounds_with_taxa.add(qid)

        return compounds_with_taxa, inchikey_map

    def fetch_all_data(endpoint: str, progress_callback=None) -> LOTUSData:
        """Fetch all required data from Wikidata."""
        queries = [
            (
                "compound_taxon",
                QUERY_COMPOUND_INCHIKEY_TAXON,
                "Fetching compound-InChIKey-taxon triplets...",
            ),
            ("taxon_ncbi", QUERY_TAXON_NCBI, "Fetching taxon-NCBI pairs..."),
            ("taxon_name", QUERY_TAXON_NAME, "Fetching taxon-name pairs..."),
            (
                "taxon_parent",
                QUERY_TAXON_PARENT,
                "Fetching taxon-parent pairs (under Biota)...",
            ),
            (
                "compound_smiles_can",
                QUERY_COMPOUND_SMILES_CAN,
                "Fetching compound-canonical SMILES pairs...",
            ),
            (
                "compound_smiles_iso",
                QUERY_COMPOUND_SMILES_ISO,
                "Fetching compound-isomeric SMILES pairs...",
            ),
            (
                "compound_smarts",
                QUERY_COMPOUND_SMARTS,
                "Fetching compound-SMARTS pairs...",
            ),
            (
                "compound_cxsmiles",
                QUERY_COMPOUND_CXSMILES,
                "Fetching compound-CXSMILES pairs...",
            ),
            (
                "compound_label",
                QUERY_COMPOUND_LABEL,
                "Fetching compound-label pairs...",
            ),
            (
                "compound_parent",
                QUERY_COMPOUND_PARENT,
                "Fetching compound-parent pairs (under chemical compound)...",
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
        """Process compound-label pairs. Prefers 'mul' labels over 'en' when both exist."""
        return (
            lf.pipe(extract_qids_from_lazyframe, "compound")
            # Sort so 'mul' comes before 'en' (alphabetically 'm' < 'e' is false, so we reverse)
            .sort("compound", "lang", descending=[False, True])
            # Keep first occurrence per compound (will be 'mul' if exists, else 'en')
            .unique(subset=["compound"], keep="first")
            .drop("lang")
        )

    def process_compound_smiles(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
        """Process compound SMILES pairs."""
        return lf.pipe(extract_qids_from_lazyframe, "compound")

    def combine_smiles(
        smiles_iso: pl.LazyFrame,
        smiles_can: pl.LazyFrame,
    ) -> pl.DataFrame:
        """
        Combine isomeric and canonical SMILES, preferring isomeric when both exist.

        Returns DataFrame with columns: compound, smiles (as list)
        """
        iso_df = smiles_iso.collect()
        can_df = smiles_can.collect()

        # Handle empty dataframes
        if len(iso_df) == 0 and len(can_df) == 0:
            return pl.DataFrame({"compound": [], "smiles": []})

        # Aggregate all SMILES per compound into lists
        if len(iso_df) > 0:
            iso_grouped = iso_df.group_by("compound").agg(
                pl.col("compound_smiles_iso").alias("smiles_iso"),
            )
        else:
            iso_grouped = pl.DataFrame({"compound": [], "smiles_iso": []})

        if len(can_df) > 0:
            can_grouped = can_df.group_by("compound").agg(
                pl.col("compound_smiles_can").alias("smiles_can"),
            )
        else:
            can_grouped = pl.DataFrame({"compound": [], "smiles_can": []})

        # Join and prefer isomeric, fall back to canonical
        combined = (
            can_grouped.join(iso_grouped, on="compound", how="full")
            .with_columns(
                pl.coalesce(["smiles_iso", "smiles_can"]).alias("smiles"),
            )
            .select(["compound", "smiles"])
        )

        return combined

    def build_structure_maps(
        smiles_iso: pl.LazyFrame,
        smiles_can: pl.LazyFrame,
        smarts: pl.LazyFrame,
        cxsmiles: pl.LazyFrame,
    ) -> tuple[dict, dict, dict]:
        """
        Build mappings for SMILES, SMARTS, and CXSMILES.
        Each mapping returns lists of values per compound.

        Returns (smiles_map, smarts_map, cxsmiles_map)
        """
        # Combine SMILES (prefer isomeric over canonical)
        smiles_df = combine_smiles(smiles_iso, smiles_can)
        smiles_map = (
            dict(
                zip(
                    smiles_df["compound"].to_list(),
                    smiles_df["smiles"].to_list(),
                ),
            )
            if len(smiles_df) > 0
            else {}
        )

        # Build SMARTS mapping (as list)
        smarts_collected = smarts.collect()
        if len(smarts_collected) > 0:
            smarts_df = smarts_collected.group_by("compound").agg(
                pl.col("compound_smarts").alias("smarts"),
            )
            smarts_map = dict(
                zip(
                    smarts_df["compound"].to_list(),
                    smarts_df["smarts"].to_list(),
                ),
            )
        else:
            smarts_map = {}

        # Build CXSMILES mapping (as list)
        cxsmiles_collected = cxsmiles.collect()
        if len(cxsmiles_collected) > 0:
            cxsmiles_df = cxsmiles_collected.group_by("compound").agg(
                pl.col("compound_cxsmiles").alias("cxsmiles"),
            )
            cxsmiles_map = dict(
                zip(
                    cxsmiles_df["compound"].to_list(),
                    cxsmiles_df["cxsmiles"].to_list(),
                ),
            )
        else:
            cxsmiles_map = {}

        return smiles_map, smarts_map, cxsmiles_map

    def build_descriptor_map(
        smiles_map: dict,
        smarts_map: dict,
        cxsmiles_map: dict,
        inchikey_map: dict[str, list[str]] = None,
    ) -> dict[str, Descriptors]:
        """
        Build a unified mapping from compound QID to Descriptors.

        Includes ALL compounds with any descriptor data (SMILES, SMARTS, CXSMILES),
        not just those with InChIKeys. This enables efficient single-lookup for
        descriptors in tree building.

        Args:
            smiles_map: Compound QID -> SMILES list
            smarts_map: Compound QID -> SMARTS list
            cxsmiles_map: Compound QID -> CXSMILES list
            inchikey_map: Optional compound QID -> InChIKey list (for compounds with taxa)

        Returns:
            dict mapping compound QID to Descriptors object
        """
        inchikey_map = inchikey_map or {}

        # Collect all compound QIDs with any descriptor
        all_qids = (
            set(smiles_map.keys())
            | set(smarts_map.keys())
            | set(cxsmiles_map.keys())
            | set(inchikey_map.keys())
        )

        descriptor_map: dict[str, Descriptors] = {}
        for qid in all_qids:
            inchikeys = inchikey_map.get(qid)
            smiles = smiles_map.get(qid)
            smarts = smarts_map.get(qid)
            cxsmiles = cxsmiles_map.get(qid)

            # Only create Descriptors if at least one field has data
            if inchikeys or smiles or smarts or cxsmiles:
                descriptor_map[qid] = Descriptors(
                    inchikey=inchikeys,
                    smiles=smiles,
                    smarts=smarts,
                    cxsmiles=cxsmiles,
                )

        return descriptor_map

    # ========================================================================
    # TREE BUILDING
    # ========================================================================

    def build_biological_tree(
        compound_taxon: pl.DataFrame,
        taxon_ncbi: pl.DataFrame,
        taxon_parent: pl.DataFrame,
        taxon_name: pl.DataFrame,
        compounds_with_taxa: set[str],
        descriptor_map: dict[str, Descriptors],
        label_map: dict[str, str],
    ) -> list[dict]:
        """
        Build biological tree JSON for PubChem.

        This tree organizes chemical compounds by their biological source taxa.
        Only includes taxa that have compounds with InChIKeys directly or in their descendants.
        All taxa are constrained to be under Biota (Q2382443).

        Args:
            compound_taxon: DataFrame with compound-taxon relationships
            taxon_ncbi: DataFrame with taxon-NCBI ID mappings
            taxon_parent: DataFrame with taxon parent relationships
            taxon_name: DataFrame with taxon names
            compounds_with_taxa: Set of compound QIDs that have InChIKeys+taxa
            descriptor_map: Unified mapping of compound QID to Descriptors
            label_map: Mapping of compound QID to label

        Returns a hierarchical tree structure where each taxon node contains:
        - Name: Taxonomic name
        - Identifiers: NCBI TaxID and Wikidata QID
        - Compounds: List of compounds found in this taxon
        - Children: List of child taxon nodes
        """
        # Step 1: Get all taxa that directly have compounds
        taxa_with_compounds = set(compound_taxon["taxon"].unique().to_list())

        # Step 2: Build child->parent mapping for upward traversal
        child_to_parent = dict(
            zip(
                taxon_parent["taxon"].to_list(),
                taxon_parent["taxon_parent"].to_list(),
            ),
        )

        # Step 3: Find all ancestors of taxa with compounds (these form our relevant tree)
        relevant_taxa = set(taxa_with_compounds)
        for taxon in taxa_with_compounds:
            current = taxon
            while current in child_to_parent:
                parent = child_to_parent[current]
                if parent in relevant_taxa:
                    break  # Already processed this branch
                relevant_taxa.add(parent)
                current = parent

        # Step 4: Filter taxon_parent to only include relevant taxa
        filtered_taxon_parent = taxon_parent.filter(
            pl.col("taxon").is_in(relevant_taxa)
            & pl.col("taxon_parent").is_in(relevant_taxa),
        )

        # Build parent-child relationships (only for relevant taxa)
        parent_map_data = (
            filtered_taxon_parent.group_by("taxon_parent")
            .agg(
                pl.col("taxon").alias("children"),
            )
            .to_dict(as_series=False)
        )
        parent_map = dict(
            zip(parent_map_data["taxon_parent"], parent_map_data["children"]),
        )

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

        # Build taxon to compound QIDs mapping
        compound_qids_by_taxon = (
            compound_taxon.group_by("taxon")
            .agg(pl.col("compound").unique().alias("compound_qids"))
            .to_dict(as_series=False)
        )
        taxon_to_compound_qids = dict(
            zip(
                compound_qids_by_taxon["taxon"],
                compound_qids_by_taxon["compound_qids"],
            ),
        )

        # Find root taxa (those in relevant_taxa without parents in our filtered set)
        all_taxa_in_filtered = set(filtered_taxon_parent["taxon"].to_list())
        all_parents_in_filtered = set(filtered_taxon_parent["taxon_parent"].to_list())
        roots = (all_parents_in_filtered - all_taxa_in_filtered) & relevant_taxa

        def build_node(taxon_qid: str, visited: set) -> dict | None:
            """Build node, returning None if no compounds in subtree."""
            if taxon_qid in visited:
                return None
            visited.add(taxon_qid)

            # Get compounds for this taxon
            compound_qids = taxon_to_compound_qids.get(taxon_qid, [])
            compounds_list = []
            for qid in compound_qids:
                if qid not in compounds_with_taxa:
                    continue  # Skip compounds without valid InChIKeys
                # Build compound dict from descriptor_map and label_map
                label = label_map.get(qid, qid)
                identifiers = Identifiers(wikidata_qid=qid)
                compound_dict = {
                    "Name": label,
                    "Identifiers": identifiers.to_dict(),
                }
                descriptors = descriptor_map.get(qid)
                if descriptors:
                    desc_dict = descriptors.to_dict()
                    if desc_dict:
                        compound_dict["Descriptors"] = desc_dict
                compounds_list.append(compound_dict)

            # Build children (only for children in relevant_taxa)
            children = parent_map.get(taxon_qid, [])
            child_nodes = []
            for child in children:
                child_node = build_node(child, visited)
                if child_node:
                    child_nodes.append(child_node)

            # Only include this node if it has compounds or has valid children
            if not compounds_list and not child_nodes:
                return None

            # Build Taxon object with identifiers
            ncbi_id = ncbi_map.get(taxon_qid)
            name = name_map.get(taxon_qid, taxon_qid)
            identifiers = Identifiers(wikidata_qid=taxon_qid, ncbi_taxid=ncbi_id)
            taxon = Taxon(identifiers=identifiers, name=name)

            # Sort compounds and children by Name for consistent output
            sorted_compounds = (
                sorted(compounds_list, key=lambda x: x.get("Name", ""))
                if compounds_list
                else None
            )
            sorted_children = (
                sorted(child_nodes, key=lambda x: x.get("Name", ""))
                if child_nodes
                else None
            )

            return taxon.to_node_dict(
                compounds=sorted_compounds,
                children=sorted_children,
            )

        # Build tree from roots
        tree = []
        visited = set()
        for root in sorted(roots):
            node = build_node(root, visited)
            if node:
                tree.append(node)

        # Sort root nodes by Name
        return sorted(tree, key=lambda x: x.get("Name", ""))

    def build_compound_tree(
        compound_parent: pl.DataFrame,
        compound_label: pl.DataFrame,
        compounds_with_taxa: set[str],
        descriptor_map: dict[str, Descriptors],
    ) -> list[dict]:
        """
        Build chemical compound tree JSON for PubChem.

        Includes:
        - Nodes with InChIKeys that are associated with taxa (from compounds_with_taxa)
        - Intermediary nodes without InChIKeys but with valid children
        - Descriptors from descriptor_map for ALL nodes when available

        All compounds are constrained to be under chemical compound (Q11173).

        Args:
            compound_parent: DataFrame with compound-parent relationships
            compound_label: DataFrame with compound labels
            compounds_with_taxa: Set of compound QIDs that have InChIKeys AND taxa
            descriptor_map: Unified mapping of compound QID to Descriptors

        Returns a hierarchical tree structure where each node contains:
        - Name: Compound label/name
        - Identifiers: Wikidata QID
        - Descriptors: Structure descriptors (InChIKey, SMILES, SMARTS, CXSMILES)
        - Children: List of child compound classes
        """
        # Build parent-child relationships
        parent_map_data = (
            compound_parent.group_by("compound_parent")
            .agg(
                pl.col("compound").alias("children"),
            )
            .to_dict(as_series=False)
        )
        parent_map = dict(
            zip(parent_map_data["compound_parent"], parent_map_data["children"]),
        )

        # Build compound label mapping
        label_map = dict(
            zip(
                compound_label["compound"].to_list(),
                compound_label["compound_label"].to_list(),
            ),
        )

        # Find root compounds
        all_compounds = set(compound_parent["compound"].to_list())
        all_parents = set(compound_parent["compound_parent"].to_list())
        roots = all_parents - all_compounds

        def build_node(compound_qid: str, visited: set) -> dict | None:
            """Build node, returning None if no valid descendants."""
            if compound_qid in visited:
                return None
            visited.add(compound_qid)

            # Check if this compound has taxa (required for inclusion)
            has_taxa = compound_qid in compounds_with_taxa

            # Build children first to check if any have valid descendants
            children = parent_map.get(compound_qid, [])
            child_nodes = []
            for child in children:
                child_node = build_node(child, visited)
                if child_node:
                    child_nodes.append(child_node)

            # Include this node ONLY if:
            # - It has InChIKeys+taxa, OR
            # - It has valid children (making it an intermediary node)
            if not has_taxa and not child_nodes:
                return None

            label = label_map.get(compound_qid, compound_qid)
            identifiers = Identifiers(wikidata_qid=compound_qid)

            # Build node with Name first for readability
            node = {
                "Name": label,
                "Identifiers": identifiers.to_dict(),
            }

            # Add descriptors from unified map (single lookup)
            descriptors = descriptor_map.get(compound_qid)
            if descriptors:
                desc_dict = descriptors.to_dict()
                if desc_dict:
                    node["Descriptors"] = desc_dict

            if child_nodes:
                # Sort children by Name
                node["Children"] = sorted(child_nodes, key=lambda x: x.get("Name", ""))

            return node

        # Build tree from roots
        tree = []
        visited = set()
        for root in sorted(roots):
            node = build_node(root, visited)
            if node:
                tree.append(node)

        # Sort root nodes by Name
        return sorted(tree, key=lambda x: x.get("Name", ""))

    def tree_to_display(tree: list[dict]) -> tuple[dict, int, int]:
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

            # Node structure: {"Name": ..., "Identifiers": ..., ...}
            name = node.get("Name", "Unknown")

            # Count compounds (biological tree) or check if "has descriptors" (chemical tree)
            n_compounds = len(node.get("Compounds", []))
            has_descriptors = "Descriptors" in node

            display_label = name
            if n_compounds > 0:
                display_label += f" ({n_compounds} compounds)"
            elif has_descriptors:
                display_label += " (has descriptors)"

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
    1. **Biological Tree**: Biological taxonomy (under [Biota](https://www.wikidata.org/wiki/Q2382443)) with associated descriptors
    2. **Chemical Tree**: Chemical taxonomy (under [chemical compound](https://www.wikidata.org/wiki/Q11173)) with associated descriptors

    *Only nodes with InChIKey and taxon association (directly or in descendants) are included.*
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
            compound_smiles_can=process_compound_smiles(
                data.compound_smiles_can,
                "compound_smiles_can",
            ),
            compound_smiles_iso=process_compound_smiles(
                data.compound_smiles_iso,
                "compound_smiles_iso",
            ),
            compound_smarts=process_compound_smiles(
                data.compound_smarts,
                "compound_smarts",
            ),
            compound_cxsmiles=process_compound_smiles(
                data.compound_cxsmiles,
                "compound_cxsmiles",
            ),
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

    with mo.status.spinner("Building structure maps..."):
        smiles_map, smarts_map, cxsmiles_map = build_structure_maps(
            data.compound_smiles_iso,
            data.compound_smiles_can,
            data.compound_smarts,
            data.compound_cxsmiles,
        )

    with mo.status.spinner("Building compound label map..."):
        compound_label_df = data.compound_label.collect()
        label_map = dict(
            zip(
                compound_label_df["compound"].to_list(),
                compound_label_df["compound_label"].to_list(),
            ),
        )

    with mo.status.spinner("Identifying compounds with taxa..."):
        # Get compounds that have InChIKeys AND are associated with taxa
        compounds_with_taxa, inchikey_map = build_compounds_with_taxa(compound_taxon_df)

    with mo.status.spinner("Building unified descriptor map..."):
        # Build unified descriptor map for ALL compounds (including intermediary nodes)
        descriptor_map = build_descriptor_map(
            smiles_map,
            smarts_map,
            cxsmiles_map,
            inchikey_map,
        )

    with mo.status.spinner("Building biological tree..."):
        taxon_ncbi_df = data.taxon_ncbi.collect()
        taxon_parent_df = data.taxon_parent.collect()
        taxon_name_df = data.taxon_name.collect()

        biological_tree = build_biological_tree(
            compound_taxon_df,
            taxon_ncbi_df,
            taxon_parent_df,
            taxon_name_df,
            compounds_with_taxa,
            descriptor_map,
            label_map,
        )

    with mo.status.spinner("Building chemical tree..."):
        compound_parent_df = data.compound_parent.collect()

        chemical_tree = build_compound_tree(
            compound_parent_df,
            compound_label_df,
            compounds_with_taxa,
            descriptor_map,
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

    bio_display, bio_shown, bio_total = tree_to_display(biological_tree)
    chem_display, chem_shown, chem_total = tree_to_display(chemical_tree)

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
    generated_at = datetime.now().isoformat()

    def build_tree_output(tree_type: str, tree: list[dict]) -> dict:
        """Build complete output with rich metadata."""
        is_biological = tree_type == "biological"

        return {
            "_documentation": {
                "overview": (
                    "This JSON contains a hierarchical tree of biological taxa with their associated natural product compounds."
                    if is_biological
                    else "This JSON contains a hierarchical tree of chemical compound classes with structural descriptors."
                ),
                "structure": {
                    "tree": "Array of root nodes. Each node is an object with Name, Identifiers, and optional Children.",
                    "node_fields": {
                        "Name": "Human-readable name (taxon name or compound label)",
                        "Identifiers": "External database identifiers (Wikidata QID, NCBI TaxID for taxa)",
                        "Compounds": "(Biological tree only) Array of compounds found in this taxon",
                        "Descriptors": "(Chemical tree only) Chemical structure representations",
                        "Children": "Array of child nodes (same structure, recursive)",
                    },
                    "descriptors_fields": {
                        "InChIKey": "IUPAC International Chemical Identifier Key",
                        "SMILES": "Simplified Molecular Input Line Entry System (isomeric preferred over canonical)",
                        "SMARTS": "SMILES Arbitrary Target Specification (substructure patterns)",
                        "CXSMILES": "ChemAxon Extended SMILES",
                    },
                },
                "notes": [
                    "Descriptor values are strings when single, arrays when multiple values exist (very rare).",
                    "All nodes are sorted alphabetically by Name.",
                    "Only nodes with InChIKey associations (directly or via descendants) are included.",
                    "Data queried from Wikidata via QLever SPARQL endpoint.",
                ],
            },
            "metadata": {
                "name": f"LOTUS {tree_type.title()} Tree",
                "description": (
                    "Hierarchical taxonomy of biological organisms with associated natural product compounds"
                    if is_biological
                    else "Hierarchical classification of chemical compounds with structural descriptors"
                ),
                "version": CONFIG["app_version"],
                "generated": generated_at,
                "generator": CONFIG["app_name"],
                "source": {
                    "name": METADATA["project"],
                    "url": METADATA["project_url"],
                    "wikidata_item": METADATA["wikidata_item"],
                    "endpoint": CONFIG["qlever_endpoint"],
                },
                "license": {
                    "data": METADATA["license_data"],
                    "code": METADATA["license_code"],
                },
                "constraints": {
                    "root": "Biota (Q2382443)"
                    if is_biological
                    else "chemical compound (Q11173)",
                    "sparql_pattern": METADATA["constraints"][
                        "biological_tree" if is_biological else "chemical_tree"
                    ],
                },
                "statistics": {
                    "root_nodes": len(tree),
                    "total_nodes": count_tree_nodes(tree),
                },
            },
            "tree": tree,
        }

    biological_output = build_tree_output("biological", biological_tree)
    chemical_output = build_tree_output("chemical", chemical_tree)

    biological_json = json.dumps(biological_output, indent=2)
    chemical_json = json.dumps(chemical_output, indent=2)

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
            "--verbose",
            "-v",
            action="store_true",
            help="Verbose output",
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
                compound_smiles_can=process_compound_smiles(
                    data.compound_smiles_can,
                    "compound_smiles_can",
                ),
                compound_smiles_iso=process_compound_smiles(
                    data.compound_smiles_iso,
                    "compound_smiles_iso",
                ),
                compound_smarts=process_compound_smiles(
                    data.compound_smarts,
                    "compound_smarts",
                ),
                compound_cxsmiles=process_compound_smiles(
                    data.compound_cxsmiles,
                    "compound_cxsmiles",
                ),
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
                print("\nBuilding structure maps...", file=sys.stderr)

            smiles_map, smarts_map, cxsmiles_map = build_structure_maps(
                data.compound_smiles_iso,
                data.compound_smiles_can,
                data.compound_smarts,
                data.compound_cxsmiles,
            )

            compound_taxon_df = data.compound_taxon.collect()
            compound_label_df = data.compound_label.collect()

            # Build label map
            label_map = dict(
                zip(
                    compound_label_df["compound"].to_list(),
                    compound_label_df["compound_label"].to_list(),
                ),
            )

            if args.verbose:
                print("\nIdentifying compounds with taxa...", file=sys.stderr)

            # Get compounds that have InChIKeys AND are associated with taxa
            compounds_with_taxa, inchikey_map = build_compounds_with_taxa(
                compound_taxon_df,
            )

            if args.verbose:
                print(
                    f"  Compounds with InChIKey+taxa: {len(compounds_with_taxa):,}",
                    file=sys.stderr,
                )

            if args.verbose:
                print("\nBuilding unified descriptor map...", file=sys.stderr)

            # Build unified descriptor map for ALL compounds
            descriptor_map = build_descriptor_map(
                smiles_map,
                smarts_map,
                cxsmiles_map,
                inchikey_map,
            )

            if args.verbose:
                print(
                    f"  Compounds with descriptors: {len(descriptor_map):,}",
                    file=sys.stderr,
                )

            if args.verbose:
                print("\nBuilding biological tree...", file=sys.stderr)

            taxon_ncbi_df = data.taxon_ncbi.collect()
            taxon_parent_df = data.taxon_parent.collect()
            taxon_name_df = data.taxon_name.collect()

            biological_tree = build_biological_tree(
                compound_taxon_df,
                taxon_ncbi_df,
                taxon_parent_df,
                taxon_name_df,
                compounds_with_taxa,
                descriptor_map,
                label_map,
            )

            if args.verbose:
                bio_nodes = count_tree_nodes(biological_tree)
                print(f"  Root nodes: {len(biological_tree)}", file=sys.stderr)
                print(f"  Total nodes: {bio_nodes:,}", file=sys.stderr)

            if args.verbose:
                print("\nBuilding chemical tree...", file=sys.stderr)

            compound_parent_df = data.compound_parent.collect()

            chemical_tree = build_compound_tree(
                compound_parent_df,
                compound_label_df,
                compounds_with_taxa,
                descriptor_map,
            )

            if args.verbose:
                chem_nodes = count_tree_nodes(chemical_tree)
                print(f"  Root nodes: {len(chemical_tree)}", file=sys.stderr)
                print(f"  Total nodes: {chem_nodes:,}", file=sys.stderr)

            date_str = datetime.now().strftime("%Y%m%d")
            generated_at = datetime.now().isoformat()

            def build_cli_tree_output(tree_type: str, tree: list[dict]) -> dict:
                """Build complete output with rich metadata for CLI."""
                is_biological = tree_type == "biological"

                return {
                    "_documentation": {
                        "overview": (
                            "This JSON contains a hierarchical tree of biological taxa with their associated natural product compounds."
                            if is_biological
                            else "This JSON contains a hierarchical tree of chemical compound classes with structural descriptors."
                        ),
                        "structure": {
                            "tree": "Array of root nodes. Each node is an object with Name, Identifiers, and optional Children.",
                            "node_fields": {
                                "Name": "Human-readable name (taxon name or compound label)",
                                "Identifiers": "External database identifiers (Wikidata QID, NCBI TaxID for taxa)",
                                "Compounds": "(Biological tree only) Array of compounds found in this taxon",
                                "Descriptors": "(Chemical tree only) Chemical structure representations",
                                "Children": "Array of child nodes (same structure, recursive)",
                            },
                            "descriptors_fields": {
                                "InChIKey": "IUPAC International Chemical Identifier Key",
                                "SMILES": "Simplified Molecular Input Line Entry System (isomeric preferred over canonical)",
                                "SMARTS": "SMILES Arbitrary Target Specification (substructure patterns)",
                                "CXSMILES": "ChemAxon Extended SMILES",
                            },
                        },
                        "notes": [
                            "Descriptor values are strings when single, arrays when multiple values exist (very rare).",
                            "All nodes are sorted alphabetically by Name.",
                            "Only nodes with InChIKey associations (directly or via descendants) are included.",
                            "Data queried from Wikidata via QLever SPARQL endpoint.",
                        ],
                    },
                    "metadata": {
                        "name": f"LOTUS {tree_type.title()} Tree",
                        "description": (
                            "Hierarchical taxonomy of biological organisms with associated natural product compounds"
                            if is_biological
                            else "Hierarchical classification of chemical compounds with structural descriptors"
                        ),
                        "version": CONFIG["app_version"],
                        "generated": generated_at,
                        "generator": CONFIG["app_name"],
                        "source": {
                            "name": METADATA["project"],
                            "url": METADATA["project_url"],
                            "wikidata_item": METADATA["wikidata_item"],
                            "endpoint": CONFIG["qlever_endpoint"],
                        },
                        "license": {
                            "data": METADATA["license_data"],
                            "code": METADATA["license_code"],
                        },
                        "constraints": {
                            "root": "Biota (Q2382443)"
                            if is_biological
                            else "chemical compound (Q11173)",
                            "sparql_pattern": METADATA["constraints"][
                                "biological_tree" if is_biological else "chemical_tree"
                            ],
                        },
                        "statistics": {
                            "root_nodes": len(tree),
                            "total_nodes": count_tree_nodes(tree),
                        },
                    },
                    "tree": tree,
                }

            biological_output = build_cli_tree_output("biological", biological_tree)
            chemical_output = build_cli_tree_output("chemical", chemical_tree)

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
