# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "polars==1.39.3",
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
    # Remote CLI export (auto-installs deps, fetches, builds, saves) - default compact PubChem format
    uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v

    # Remote CLI export with full (detailed with metadata) format
    uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v --format full

    # Remote GUI (interactive)
    uvx marimo run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py

    # Local GUI
    marimo run lotus_pubchem_tree.py

    # Local CLI export
    python lotus_pubchem_tree.py export -o ./output -v
"""

import marimo

__generated_with = "0.21.1"
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
        "npclassifier_cache_url": "https://adafede.github.io/marimo/apps/public/npclassifier/npclassifier_cache.csv",
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

    # Reference queries - fetch DOIs and PMIDs for compound-taxon relationships
    QUERY_REFERENCE_DOI = """
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?compound ?taxon ?reference ?doi WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P356 ?doi .
    }
    """

    QUERY_REFERENCE_PMID = """
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?compound ?taxon ?reference ?pmid WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P698 ?pmid .
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
        "reference_doi": {
            "compound": pl.Utf8,
            "taxon": pl.Utf8,
            "reference": pl.Utf8,
            "doi": pl.Utf8,
        },
        "reference_pmid": {
            "compound": pl.Utf8,
            "taxon": pl.Utf8,
            "reference": pl.Utf8,
            "pmid": pl.Utf8,
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
        reference_doi: pl.LazyFrame
        reference_pmid: pl.LazyFrame

    @dataclass
    class Identifiers:
        """Identifiers for an entity (compound or taxon)."""

        wikidata_qid: str
        ncbi_taxid: str | None = None

        def to_dict(self) -> dict:
            """Convert to dictionary, omitting None fields."""
            result = {"QID": self.wikidata_qid}
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
            (
                "reference_doi",
                QUERY_REFERENCE_DOI,
                "Fetching reference DOIs...",
            ),
            (
                "reference_pmid",
                QUERY_REFERENCE_PMID,
                "Fetching reference PMIDs...",
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

    def process_reference_doi(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process reference DOI data."""
        return (
            lf.pipe(extract_qids_from_lazyframe, "compound")
            .pipe(extract_qids_from_lazyframe, "taxon")
            .pipe(extract_qids_from_lazyframe, "reference")
        )

    def process_reference_pmid(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process reference PMID data."""
        return (
            lf.pipe(extract_qids_from_lazyframe, "compound")
            .pipe(extract_qids_from_lazyframe, "taxon")
            .pipe(extract_qids_from_lazyframe, "reference")
        )

    def build_reference_map(
        reference_doi_df: pl.DataFrame,
        reference_pmid_df: pl.DataFrame,
    ) -> dict[str, dict[str, dict]]:
        """
        Build a nested mapping: compound -> taxon -> {reference_qid: {doi, pmid}}.

        This allows looking up references for a specific compound-taxon pair.
        """
        reference_map: dict[str, dict[str, dict]] = {}

        # Process DOI references
        if len(reference_doi_df) > 0:
            for row in reference_doi_df.iter_rows(named=True):
                compound = row["compound"]
                taxon = row["taxon"]
                ref_qid = row["reference"]
                doi = row["doi"]

                if compound not in reference_map:
                    reference_map[compound] = {}
                if taxon not in reference_map[compound]:
                    reference_map[compound][taxon] = {}
                if ref_qid not in reference_map[compound][taxon]:
                    reference_map[compound][taxon][ref_qid] = {}
                reference_map[compound][taxon][ref_qid]["DOI"] = doi

        # Process PMID references
        if len(reference_pmid_df) > 0:
            for row in reference_pmid_df.iter_rows(named=True):
                compound = row["compound"]
                taxon = row["taxon"]
                ref_qid = row["reference"]
                pmid = row["pmid"]

                if compound not in reference_map:
                    reference_map[compound] = {}
                if taxon not in reference_map[compound]:
                    reference_map[compound][taxon] = {}
                if ref_qid not in reference_map[compound][taxon]:
                    reference_map[compound][taxon][ref_qid] = {}
                reference_map[compound][taxon][ref_qid]["PMID"] = pmid

        return reference_map

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
        del smiles_df

        # Build SMARTS mapping (as list)
        smarts_has_data = len(smarts.collect_schema()) > 0
        if smarts_has_data:
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
                del smarts_df
            else:
                smarts_map = {}
            del smarts_collected
        else:
            smarts_map = {}

        # Build CXSMILES mapping (as list)
        cxsmiles_has_data = len(cxsmiles.collect_schema()) > 0
        if cxsmiles_has_data:
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
                del cxsmiles_df
            else:
                cxsmiles_map = {}
            del cxsmiles_collected
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
        reference_map: dict[str, dict[str, dict]] | None = None,
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
            reference_map: Nested mapping compound -> taxon -> {ref_qid: {DOI, PMID}}

        Returns a hierarchical tree structure where each taxon node contains:
        - Name: Taxonomic name
        - Identifiers: NCBI TaxID and Wikidata QID
        - Compounds: List of compounds found in this taxon (with optional References)
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
                # Add references for this specific compound-taxon pair
                if reference_map and qid in reference_map:
                    taxon_refs = reference_map[qid].get(taxon_qid)
                    if taxon_refs:
                        refs_list = []
                        for ref_qid, ref_data in taxon_refs.items():
                            ref_entry = {"QID": ref_qid}
                            ref_entry.update(ref_data)
                            refs_list.append(ref_entry)
                        if refs_list:
                            # Sort references by DOI then PMID for consistent output
                            compound_dict["References"] = sorted(
                                refs_list,
                                key=lambda r: (r.get("DOI", ""), r.get("PMID", "")),
                            )
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

    # ========================================================================
    # NPCLASSIFIER CHEMICAL TREE
    # ========================================================================
    # NPClassifier provides an alternative chemical classification based on
    # pathway → superclass → class hierarchy. This is more accepted than
    # Wikidata's P279 (subclass of) relationships for natural products.
    # ========================================================================

    def fetch_npclassifier_cache(url: str = None) -> pl.DataFrame:
        """
        Fetch NPClassifier cache CSV from remote URL.

        The cache contains SMILES with their NPClassifier annotations:
        - pathway: Top-level classification (e.g., "Terpenoids", "Alkaloids")
        - superclass: Mid-level (e.g., "Sesquiterpenoids")
        - class: Specific class (e.g., "Germacrane sesquiterpenoids")

        Multiple values are separated by " $ ".
        """
        import urllib.request

        url = url or CONFIG["npclassifier_cache_url"]
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                csv_bytes = resp.read()
            return pl.read_csv(
                io.BytesIO(csv_bytes),
                schema_overrides={
                    "smiles": pl.Utf8,
                    "pathway": pl.Utf8,
                    "superclass": pl.Utf8,
                    "class": pl.Utf8,
                    "isglycoside": pl.Utf8,
                    "error": pl.Utf8,
                },
            )
        except Exception as e:
            print(f"Warning: Could not fetch NPClassifier cache: {e}", file=sys.stderr)
            return pl.DataFrame()

    def build_npclassifier_tree(
        npclassifier_df: pl.DataFrame,
        smiles_to_inchikey: dict[str, list[str]],
        smiles_to_qid: dict[str, str] | None = None,
        compounds_with_taxa: set[str] | None = None,
    ) -> list[dict]:
        """
        Build chemical tree based on NPClassifier hierarchy.

        The NPClassifier hierarchy is: pathway → superclass → class → InChIKey

        Args:
            npclassifier_df: DataFrame with SMILES and NPClassifier annotations
            smiles_to_inchikey: Mapping from SMILES to list of InChIKeys
            smiles_to_qid: Mapping from SMILES to Wikidata QID
            compounds_with_taxa: Optional set of compound QIDs with taxa (not used here)

        Returns:
            Hierarchical tree structure compatible with the existing format
        """
        smiles_to_qid = smiles_to_qid or {}

        if len(npclassifier_df) == 0:
            return []

        # Filter out rows with errors or empty classifications
        valid_df = npclassifier_df.filter(
            (pl.col("error").is_null() | (pl.col("error") == ""))
            & (pl.col("pathway").is_not_null())
            & (pl.col("pathway") != ""),
        )

        if len(valid_df) == 0:
            return []

        # Build the tree structure: pathway -> superclass -> class -> SMILES
        # Using nested dicts for efficient building
        tree_data: dict = {}  # pathway -> superclass -> class -> set[smiles]

        for row in valid_df.iter_rows(named=True):
            smiles = row["smiles"]
            pathways = row["pathway"].split(" $ ") if row["pathway"] else []
            superclasses = row["superclass"].split(" $ ") if row["superclass"] else []
            classes = row["class"].split(" $ ") if row["class"] else []

            # Handle multiple classifications per SMILES
            # Create all pathway-superclass-class combinations
            for pathway in pathways:
                pathway = pathway.strip()
                if not pathway:
                    continue
                if pathway not in tree_data:
                    tree_data[pathway] = {}

                # Match superclasses and classes (they should correspond)
                for i, superclass in enumerate(superclasses):
                    superclass = superclass.strip()
                    if not superclass:
                        continue
                    if superclass not in tree_data[pathway]:
                        tree_data[pathway][superclass] = {}

                    # Get corresponding class (if exists)
                    if i < len(classes):
                        cls = classes[i].strip()
                        if cls:
                            if cls not in tree_data[pathway][superclass]:
                                tree_data[pathway][superclass][cls] = set()
                            tree_data[pathway][superclass][cls].add(smiles)
                    else:
                        # No class, add directly to superclass
                        if "_unclassified" not in tree_data[pathway][superclass]:
                            tree_data[pathway][superclass]["_unclassified"] = set()
                        tree_data[pathway][superclass]["_unclassified"].add(smiles)

        # Convert to the standard tree format
        tree = []
        for pathway in sorted(tree_data.keys()):
            pathway_node = {
                "Name": pathway,
                "Identifiers": {"NPClassifier_Pathway": pathway},
                "Children": [],
            }

            for superclass in sorted(tree_data[pathway].keys()):
                superclass_node = {
                    "Name": superclass,
                    "Identifiers": {"NPClassifier_Superclass": superclass},
                    "Children": [],
                }

                for cls in sorted(tree_data[pathway][superclass].keys()):
                    if cls == "_unclassified":
                        # Add SMILES directly to superclass
                        smiles_set = tree_data[pathway][superclass][cls]
                        for smi in sorted(smiles_set):
                            inchikeys = smiles_to_inchikey.get(smi, [])
                            qid = smiles_to_qid.get(smi)
                            if inchikeys:
                                for ik in inchikeys:
                                    node_identifiers = {"QID": qid} if qid else {}
                                    superclass_node["Children"].append(
                                        {
                                            "Name": ik,
                                            "Identifiers": node_identifiers,
                                            "Descriptors": {
                                                "InChIKey": ik,
                                                "SMILES": smi,
                                            },
                                        },
                                    )
                    else:
                        class_node = {
                            "Name": cls,
                            "Identifiers": {"NPClassifier_Class": cls},
                            "Children": [],
                        }
                        smiles_set = tree_data[pathway][superclass][cls]
                        for smi in sorted(smiles_set):
                            inchikeys = smiles_to_inchikey.get(smi, [])
                            qid = smiles_to_qid.get(smi)
                            if inchikeys:
                                for ik in inchikeys:
                                    node_identifiers = {"QID": qid} if qid else {}
                                    class_node["Children"].append(
                                        {
                                            "Name": ik,
                                            "Identifiers": node_identifiers,
                                            "Descriptors": {
                                                "InChIKey": ik,
                                                "SMILES": smi,
                                            },
                                        },
                                    )
                        if class_node["Children"]:
                            superclass_node["Children"].append(class_node)

                if superclass_node["Children"]:
                    pathway_node["Children"].append(superclass_node)

            if pathway_node["Children"]:
                tree.append(pathway_node)

        return tree

    def build_smiles_to_inchikey_map(
        compound_taxon_df: pl.DataFrame,
        smiles_map: dict[str, list[str]],
        inchikey_map: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """
        Build mappings from SMILES to InChIKeys and SMILES to Wikidata QID.

        This allows linking NPClassifier data (keyed by SMILES) to
        InChIKeys and Wikidata QIDs for the final tree output.

        Returns:
            Tuple of (smiles_to_inchikey, smiles_to_qid)
        """
        smiles_to_inchikey: dict[str, list[str]] = {}
        smiles_to_qid: dict[str, str] = {}

        # For each compound, map its SMILES to its InChIKeys and QID
        for qid, smiles_list in smiles_map.items():
            inchikeys = inchikey_map.get(qid, [])
            if inchikeys:
                for smi in smiles_list:
                    if smi not in smiles_to_inchikey:
                        smiles_to_inchikey[smi] = []
                    for ik in inchikeys:
                        if ik not in smiles_to_inchikey[smi]:
                            smiles_to_inchikey[smi].append(ik)
                    # Also map SMILES to QID (use first QID if multiple)
                    if smi not in smiles_to_qid:
                        smiles_to_qid[smi] = qid

        return smiles_to_inchikey, smiles_to_qid

    def npclassifier_tree_to_pubchem(tree: list[dict]) -> dict:
        """
        Convert NPClassifier tree to PubChem format.

        Structure:
        {
          "children": {
            "Pathway": {
              "children": {
                "Superclass": {
                  "children": {
                    "Class": {
                      "children": {
                        "INCHIKEY-...": {"QID": "Q..."}
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        if not tree:
            return {}

        def convert_node(node: dict) -> dict:
            name = node.get("Name", "Unknown")
            identifiers = node.get("Identifiers", {})
            children = node.get("Children", [])
            descriptors = node.get("Descriptors", {})

            result = {}

            if children:
                result["children"] = {}
                for child in children:
                    child_name = child.get("Name", "Unknown")
                    result["children"][child_name] = convert_node(child)
            elif descriptors.get("InChIKey"):
                # Leaf node - include Wikidata QID if available
                qid = identifiers.get("QID")
                if qid:
                    return {"QID": qid}
                return {}

            return result

        pubchem_tree = {"children": {}}
        for node in tree:
            name = node.get("Name", "Unknown")
            pubchem_tree["children"][name] = convert_node(node)

        return pubchem_tree

    # ========================================================================
    # PUBCHEM FORMAT CONVERSION
    # ========================================================================
    # The old PubChem format uses dict-based nesting with names as keys:
    #   {"children": {"Name1": {"children": {...}, "NCBI_ID": {...}}}}
    # The internal format uses arrays:
    #   [{"Name": "...", "Children": [...]}]
    # These functions convert between formats.
    # ========================================================================

    def tree_to_pubchem_format(
        tree: list[dict],
        tree_type: str,
    ) -> dict:
        """
        Convert array-based tree to PubChem's dict-based format.

        For biological tree:
          - Names as keys for intermediate nodes
          - NCBI TaxID as key for leaf taxa with InChIKeys as children
        For chemical tree:
          - Names as keys for intermediate nodes
          - InChIKeys as leaf keys with empty arrays

        Args:
            tree: Array-based tree from build_*_tree functions
            tree_type: "biological" or "chemical"

        Returns:
            Dict-based tree in PubChem format
        """
        if tree_type == "biological":
            return convert_bio_node_to_pubchem(tree)
        else:
            return convert_chem_node_to_pubchem(tree)

    def convert_bio_node_to_pubchem(nodes: list[dict]) -> dict:
        """
        Convert biological tree nodes to PubChem format.

        Structure:
        {
          "children": {
            "TaxonName": {
              "QID": "Q...",
              "children": {
                "ChildTaxonName": {...},
                "NCBI_ID": {
                  "organism_name": ["TaxonName"],
                  "compounds": {
                    "InChIKey1": {"QID": "Q...", "references": [{"DOI": "...", "PMID": "..."}]},
                    "InChIKey2": {"QID": "Q...", "references": [...]}
                  }
                }
              }
            }
          }
        }

        Compounds are keyed by InChIKey and include their Wikidata QID and references.
        """
        if not nodes:
            return {}

        result = {"children": {}}

        for node in nodes:
            name = node.get("Name", "Unknown")
            identifiers = node.get("Identifiers", {})
            taxon_qid = identifiers.get("QID")
            ncbi_id = identifiers.get("NCBI_TaxID")
            children = node.get("Children", [])
            compounds = node.get("Compounds", [])

            # Build this node's content
            node_content = {}

            # Add taxon Wikidata QID
            if taxon_qid:
                node_content["QID"] = taxon_qid

            # Recursively convert children
            if children:
                child_result = convert_bio_node_to_pubchem(children)
                if "children" in child_result:
                    node_content["children"] = child_result["children"]

            # Add NCBI ID entry if present (this is the leaf identifier)
            if ncbi_id:
                if "children" not in node_content:
                    node_content["children"] = {}
                ncbi_entry = {"organism_name": [name]}

                # Add compounds with their InChIKeys, Wikidata QIDs, and references
                # Aggregate references when the same InChIKey appears multiple times
                if compounds:
                    compounds_dict: dict[str, dict] = {}
                    for compound in compounds:
                        compound_identifiers = compound.get("Identifiers", {})
                        compound_qid = compound_identifiers.get("QID")
                        descriptors = compound.get("Descriptors", {})
                        inchikey = descriptors.get("InChIKey")
                        references = compound.get("References", [])

                        if inchikey:
                            # Handle both single InChIKey and list of InChIKeys
                            inchikeys = (
                                [inchikey] if isinstance(inchikey, str) else inchikey
                            )
                            for ik in inchikeys:
                                # Aggregate references for this InChIKey
                                if ik not in compounds_dict:
                                    compounds_dict[ik] = {"_refs_set": set()}
                                    # Add Wikidata QID for the compound
                                    if compound_qid:
                                        compounds_dict[ik]["QID"] = compound_qid

                                if references:
                                    for ref in references:
                                        # Create a hashable key for deduplication
                                        doi = ref.get("DOI")
                                        pmid = ref.get("PMID")
                                        ref_key = (doi, pmid)

                                        if (
                                            ref_key
                                            not in compounds_dict[ik]["_refs_set"]
                                        ):
                                            compounds_dict[ik]["_refs_set"].add(ref_key)
                                            if "references" not in compounds_dict[ik]:
                                                compounds_dict[ik]["references"] = []
                                            ref_entry = {}
                                            if doi:
                                                ref_entry["DOI"] = doi
                                            if pmid:
                                                ref_entry["PMID"] = pmid
                                            if ref_entry:
                                                compounds_dict[ik]["references"].append(
                                                    ref_entry,
                                                )

                    # Clean up internal tracking sets before output
                    for ik in compounds_dict:
                        compounds_dict[ik].pop("_refs_set", None)
                        if not compounds_dict[ik]:
                            compounds_dict[ik] = {}

                    if compounds_dict:
                        ncbi_entry["compounds"] = compounds_dict

                node_content["children"][ncbi_id] = ncbi_entry

            result["children"][name] = node_content if node_content else {}

        return result

    def convert_chem_node_to_pubchem(nodes: list[dict]) -> dict:
        """
        Convert chemical tree nodes to PubChem format.

        Structure:
        {
          "children": {
            "CompoundClassName": {
              "QID": "Q...",
              "children": {
                "INCHIKEY-...": {"QID": "Q..."}
              }
            }
          }
        }
        """
        if not nodes:
            return {}

        result = {"children": {}}

        for node in nodes:
            name = node.get("Name", "Unknown")
            identifiers = node.get("Identifiers", {})
            qid = identifiers.get("QID")
            descriptors = node.get("Descriptors", {})
            children = node.get("Children", [])

            # Build this node's content
            node_content = {}

            # Add Wikidata QID for this node
            if qid:
                node_content["QID"] = qid

            # Recursively convert children
            if children:
                child_result = convert_chem_node_to_pubchem(children)
                if "children" in child_result:
                    node_content["children"] = child_result["children"]

            # If this node has an InChIKey, add it as a leaf with QID
            inchikey = descriptors.get("InChIKey")
            if inchikey:
                if isinstance(inchikey, list):
                    for ik in inchikey:
                        if "children" not in node_content:
                            node_content["children"] = {}
                        leaf_content = {"QID": qid} if qid else {}
                        node_content["children"][ik] = leaf_content
                else:
                    if "children" not in node_content:
                        node_content["children"] = {}
                    leaf_content = {"QID": qid} if qid else {}
                    node_content["children"][inchikey] = leaf_content

            result["children"][name] = node_content if node_content else {}

        return result


@app.cell
def md_title():
    mo.md("""
    # LOTUS PubChem Tree Generator

    This app generates hierarchical JSON files for PubChem classification matching
    [PubChem Classification](https://pubchem.ncbi.nlm.nih.gov/classification/#hid=115).

    It produces two JSON files:
    1. **Biological Tree**: Biological taxonomy (under [Biota](https://www.wikidata.org/wiki/Q2382443)) with associated descriptors
    2. **Chemical Tree**: Chemical taxonomy (under [chemical compound](https://www.wikidata.org/wiki/Q11173)) with associated descriptors

    *Only nodes with InChIKey and taxon association (directly or in descendants) are included.*
    """)
    return


@app.cell
def wasm_warning():
    if IS_PYODIDE:
        mo.stop(
            True,
            mo.callout(
                mo.md("""
                **This app cannot run in the browser**

                The LOTUS PubChem Tree Generator processes lasrge amount
                of data from Wikidata, which exceeds WebAssembly memory limits.

                **Please use the CLI instead:**
                ```bash
                # Full format with metadata
                uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v

                # PubChem compact format
                uv run https://adafede.github.io/marimo/apps/lotus_pubchem_tree.py export -o ./output -v --format pubchem
                ```

                The CLI runs natively and has no memory limitations.
                """),
                kind="danger",
            ),
        )
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
            reference_doi=process_reference_doi(data.reference_doi),
            reference_pmid=process_reference_pmid(data.reference_pmid),
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

    # Step 1: Collect compound_taxon first (needed for multiple steps)
    with mo.status.spinner("Loading compound-taxon data..."):
        compound_taxon_df = data.compound_taxon.collect()

    # Step 2: Build structure maps (one at a time to save memory)
    with mo.status.spinner("Building SMILES map..."):
        smiles_map, _, _ = build_structure_maps(
            data.compound_smiles_iso,
            data.compound_smiles_can,
            pl.LazyFrame(),  # Empty - we'll do SMARTS separately
            pl.LazyFrame(),  # Empty - we'll do CXSMILES separately
        )

    with mo.status.spinner("Building SMARTS map..."):
        _, smarts_map, _ = build_structure_maps(
            pl.LazyFrame(),
            pl.LazyFrame(),
            data.compound_smarts,
            pl.LazyFrame(),
        )

    with mo.status.spinner("Building CXSMILES map..."):
        _, _, cxsmiles_map = build_structure_maps(
            pl.LazyFrame(),
            pl.LazyFrame(),
            pl.LazyFrame(),
            data.compound_cxsmiles,
        )

    # Step 3: Build label map
    with mo.status.spinner("Building compound label map..."):
        compound_label_df = data.compound_label.collect()
        label_map = dict(
            zip(
                compound_label_df["compound"].to_list(),
                compound_label_df["compound_label"].to_list(),
            ),
        )

    # Step 4: Identify compounds with taxa
    with mo.status.spinner("Identifying compounds with taxa..."):
        compounds_with_taxa, inchikey_map = build_compounds_with_taxa(compound_taxon_df)

    # Step 5: Build unified descriptor map
    with mo.status.spinner("Building unified descriptor map..."):
        descriptor_map = build_descriptor_map(
            smiles_map,
            smarts_map,
            cxsmiles_map,
            inchikey_map,
        )

    # Step 5b: Build SMILES to InChIKey mapping for NPClassifier
    with mo.status.spinner("Building SMILES to InChIKey mapping..."):
        smiles_to_inchikey, smiles_to_qid = build_smiles_to_inchikey_map(
            compound_taxon_df,
            smiles_map,
            inchikey_map,
        )

    # Free the individual maps now that we have what we need
    del smarts_map, cxsmiles_map

    # Step 6: Build biological tree
    with mo.status.spinner("Loading taxon data..."):
        taxon_ncbi_df = data.taxon_ncbi.collect()
        taxon_parent_df = data.taxon_parent.collect()
        taxon_name_df = data.taxon_name.collect()

    with mo.status.spinner("Building reference map..."):
        reference_doi_df = data.reference_doi.collect()
        reference_pmid_df = data.reference_pmid.collect()
        reference_map = build_reference_map(reference_doi_df, reference_pmid_df)
        del reference_doi_df, reference_pmid_df

    with mo.status.spinner("Building biological tree..."):
        biological_tree = build_biological_tree(
            compound_taxon_df,
            taxon_ncbi_df,
            taxon_parent_df,
            taxon_name_df,
            compounds_with_taxa,
            descriptor_map,
            label_map,
            reference_map,
        )
        # Free taxon DataFrames after biological tree is built
        del taxon_ncbi_df, taxon_parent_df, taxon_name_df, reference_map

    # Step 7: Build Wikidata-based chemical tree
    with mo.status.spinner("Loading compound parent data..."):
        compound_parent_df = data.compound_parent.collect()

    with mo.status.spinner("Building Wikidata chemical tree..."):
        chemical_tree = build_compound_tree(
            compound_parent_df,
            compound_label_df,
            compounds_with_taxa,
            descriptor_map,
        )

    # Step 8: Build NPClassifier-based chemical tree
    with mo.status.spinner("Fetching NPClassifier cache..."):
        npclassifier_df = fetch_npclassifier_cache()

    npclassifier_tree = []
    if len(npclassifier_df) > 0:
        with mo.status.spinner("Building NPClassifier chemical tree..."):
            npclassifier_tree = build_npclassifier_tree(
                npclassifier_df,
                smiles_to_inchikey,
                smiles_to_qid,
            )
        del npclassifier_df
    else:
        mo.output.append(
            mo.callout(
                mo.md(
                    "Could not fetch NPClassifier cache. NPClassifier tree will not be available.",
                ),
                kind="warn",
            ),
        )

    # Free remaining DataFrames
    del compound_taxon_df, compound_parent_df, compound_label_df
    del compounds_with_taxa, descriptor_map, label_map
    del smiles_map, inchikey_map, smiles_to_inchikey, smiles_to_qid

    bio_nodes = count_tree_nodes(biological_tree)
    chem_nodes = count_tree_nodes(chemical_tree)
    npc_nodes = count_tree_nodes(npclassifier_tree) if npclassifier_tree else 0

    _output = [
        mo.md(f"""
    ## Trees Built

    - **Biological Tree**: {len(biological_tree)} root nodes, {bio_nodes:,} total nodes
    - **Chemical Tree (Wikidata)**: {len(chemical_tree)} root nodes, {chem_nodes:,} total nodes
    - **Chemical Tree (NPClassifier)**: {len(npclassifier_tree)} root nodes, {npc_nodes:,} total nodes
        """),
    ]

    mo.vstack(_output)
    return biological_tree, chemical_tree, npclassifier_tree


@app.cell
def display_previews(biological_tree, chemical_tree, npclassifier_tree):
    mo.stop(biological_tree is None or chemical_tree is None)

    bio_display, bio_shown, bio_total = tree_to_display(biological_tree)
    chem_display, chem_shown, chem_total = tree_to_display(chemical_tree)
    npc_display, npc_shown, npc_total = (
        tree_to_display(npclassifier_tree) if npclassifier_tree else ({}, 0, 0)
    )

    total_shown = bio_shown + chem_shown + npc_shown
    total_nodes = bio_total + chem_total + npc_total

    tabs_dict = {
        f"Biological Tree ({bio_total:,} nodes)": mo.tree(bio_display),
        f"Chemical Tree - Wikidata ({chem_total:,} nodes)": mo.vstack(
            [
                mo.callout(
                    mo.md("""
    **Note:** The Wikidata-based chemical tree relies on P279 (subclass of) relationships,
    which are currently sparse for natural products. Use the NPClassifier-based tree below
    for a more comprehensive chemical classification.
                """),
                    kind="warn",
                ),
                mo.tree(chem_display),
            ],
        ),
    }

    if npclassifier_tree:
        tabs_dict[f"Chemical Tree - NPClassifier ({npc_total:,} nodes)"] = mo.vstack(
            [
                mo.callout(
                    mo.md("""
    This tree uses [NPClassifier](https://npclassifier.gnps2.org/) for
    chemical classification. NPClassifier provides a comprehensive pathway → superclass → class
    hierarchy specifically designed for natural products.
                """),
                    kind="success",
                ),
                mo.tree(npc_display),
            ],
        )

    mo.vstack(
        [
            mo.callout(
                mo.md(f"""
    **Preview is truncated for performance.** Showing ~{total_shown:,} nodes out of {total_nodes:,} total.
    Download the JSON files for the complete trees.
            """),
                kind="info",
            ),
            mo.ui.tabs(tabs_dict),
        ],
    )
    return


@app.cell
def download_buttons(biological_tree, chemical_tree, npclassifier_tree):
    mo.stop(biological_tree is None or chemical_tree is None)

    date_str = datetime.now().strftime("%Y%m%d")
    generated_at = datetime.now().isoformat()

    def build_tree_output(
        tree_type: str,
        tree: list[dict],
        source: str = "wikidata",
    ) -> dict:
        """Build complete output with rich metadata."""
        is_biological = tree_type == "biological"
        is_npclassifier = source == "npclassifier"

        if is_npclassifier:
            overview = "This JSON contains a hierarchical tree of chemical compounds classified using NPClassifier (pathway → superclass → class)."
            description = (
                "NPClassifier-based hierarchical classification of natural products"
            )
            root_info = "NPClassifier pathways"
            notes = [
                "NPClassifier provides a comprehensive classification for natural products.",
                "Hierarchy: pathway → superclass → class → InChIKey",
                "Multiple classifications are possible for a single compound.",
                "See https://npclassifier.gnps2.org/ for more information.",
            ]
        elif is_biological:
            overview = "This JSON contains a hierarchical tree of biological taxa with their associated natural product compounds."
            description = "Hierarchical taxonomy of biological organisms with associated natural product compounds"
            root_info = "Biota (Q2382443)"
            notes = [
                "Descriptor values are strings when single, arrays when multiple values exist (very rare).",
                "All nodes are sorted alphabetically by Name.",
                "Only nodes with InChIKey associations (directly or via descendants) are included.",
                "Data queried from Wikidata via QLever SPARQL endpoint.",
            ]
        else:
            overview = "This JSON contains a hierarchical tree of chemical compound classes with structural descriptors."
            description = "Hierarchical classification of chemical compounds with structural descriptors"
            root_info = "chemical compound (Q11173)"
            notes = [
                "Note: Wikidata P279 (subclass of) relationships are sparse for natural products.",
                "Consider using the NPClassifier-based tree for better coverage.",
                "Descriptor values are strings when single, arrays when multiple values exist (very rare).",
                "All nodes are sorted alphabetically by Name.",
            ]

        return {
            "_documentation": {
                "overview": overview,
                "structure": {
                    "tree": "Array of root nodes. Each node is an object with Name, Identifiers, and optional Children.",
                    "node_fields": {
                        "Name": "Human-readable name (taxon name or compound label)",
                        "Identifiers": "External database identifiers (Wikidata QID, NCBI TaxID for taxa, NPClassifier levels)",
                        "Compounds": "(Biological tree only) Array of compounds found in this taxon",
                        "Descriptors": "Chemical structure representations (InChIKey, SMILES, etc.)",
                        "References": "(Biological tree compounds only) Literature references for compound-taxon association",
                        "Children": "Array of child nodes (same structure, recursive)",
                    },
                    "descriptors_fields": {
                        "InChIKey": "IUPAC International Chemical Identifier Key",
                        "SMILES": "Simplified Molecular Input Line Entry System (isomeric preferred over canonical)",
                        "SMARTS": "SMILES Arbitrary Target Specification (substructure patterns)",
                        "CXSMILES": "ChemAxon Extended SMILES",
                    },
                    "reference_fields": {
                        "QID": "QID of the reference article in Wikidata",
                        "DOI": "Digital Object Identifier of the reference",
                        "PMID": "PubMed ID of the reference",
                    },
                },
                "notes": notes,
            },
            "metadata": {
                "name": f"LOTUS {tree_type.title()} Tree"
                + (" (NPClassifier)" if is_npclassifier else ""),
                "description": description,
                "version": CONFIG["app_version"],
                "generated": generated_at,
                "generator": CONFIG["app_name"],
                "source": {
                    "name": METADATA["project"],
                    "url": METADATA["project_url"],
                    "wikidata_item": METADATA["wikidata_item"],
                    "endpoint": CONFIG["qlever_endpoint"]
                    if not is_npclassifier
                    else CONFIG["npclassifier_cache_url"],
                    "classification": "NPClassifier"
                    if is_npclassifier
                    else "Wikidata P279",
                },
                "license": {
                    "data": METADATA["license_data"],
                    "code": METADATA["license_code"],
                },
                "constraints": {
                    "root": root_info,
                },
                "statistics": {
                    "root_nodes": len(tree),
                    "total_nodes": count_tree_nodes(tree),
                },
            },
            "tree": tree,
        }

    # PubChem format (default, compact name-as-key)
    biological_pubchem = tree_to_pubchem_format(biological_tree, "biological")
    chemical_pubchem = tree_to_pubchem_format(chemical_tree, "chemical")
    biological_pubchem_json = json.dumps(biological_pubchem, indent=2)
    chemical_pubchem_json = json.dumps(chemical_pubchem, indent=2)

    # NPClassifier tree (PubChem compact format)
    npclassifier_pubchem_json = ""
    npclassifier_full_json = ""
    if npclassifier_tree:
        npclassifier_pubchem = npclassifier_tree_to_pubchem(npclassifier_tree)
        npclassifier_pubchem_json = json.dumps(npclassifier_pubchem, indent=2)
        npclassifier_output = build_tree_output(
            "chemical",
            npclassifier_tree,
            source="npclassifier",
        )
        npclassifier_full_json = json.dumps(npclassifier_output, indent=2)

    # Full format (detailed with metadata)
    biological_output = build_tree_output("biological", biological_tree)
    chemical_output = build_tree_output("chemical", chemical_tree)
    biological_full_json = json.dumps(biological_output, indent=2)
    chemical_full_json = json.dumps(chemical_output, indent=2)

    download_elements = [
        mo.md("## Download Trees"),
        mo.callout(
            mo.md("""
    **Chemical Tree Options:**
    - **Wikidata-based**: Uses P279 (subclass of) relationships from Wikidata. Currently sparse for natural products.
    - **NPClassifier-based**: Uses NPClassifier's pathway → superclass → class hierarchy, specifically designed for natural products.
            """),
            kind="info",
        ),
        mo.md("### PubChem Format (default, compact)"),
        mo.hstack(
            [
                mo.download(
                    label="Biological Tree JSON",
                    filename=f"{date_str}_lotus_biological_tree.json",
                    mimetype="application/json",
                    data=lambda: biological_pubchem_json.encode("utf-8"),
                ),
                mo.download(
                    label="Chemical Tree (Wikidata) JSON",
                    filename=f"{date_str}_lotus_chemical_tree_wikidata.json",
                    mimetype="application/json",
                    data=lambda: chemical_pubchem_json.encode("utf-8"),
                ),
            ]
            + (
                [
                    mo.download(
                        label="Chemical Tree JSON",
                        filename=f"{date_str}_lotus_chemical_tree.json",
                        mimetype="application/json",
                        data=lambda: npclassifier_pubchem_json.encode("utf-8"),
                    ),
                ]
                if npclassifier_tree
                else []
            ),
            gap=2,
        ),
        mo.md("### Full Format (detailed with metadata)"),
        mo.hstack(
            [
                mo.download(
                    label="Biological Tree (Full)",
                    filename=f"{date_str}_lotus_biological_tree_full.json",
                    mimetype="application/json",
                    data=lambda: biological_full_json.encode("utf-8"),
                ),
                mo.download(
                    label="Chemical Tree Wikidata (Full)",
                    filename=f"{date_str}_lotus_chemical_tree_wikidata_full.json",
                    mimetype="application/json",
                    data=lambda: chemical_full_json.encode("utf-8"),
                ),
            ]
            + (
                [
                    mo.download(
                        label="Chemical Tree (Full)",
                        filename=f"{date_str}_lotus_chemical_tree_full.json",
                        mimetype="application/json",
                        data=lambda: npclassifier_full_json.encode("utf-8"),
                    ),
                ]
                if npclassifier_tree
                else []
            ),
            gap=2,
        ),
    ]

    mo.vstack(download_elements)
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
  uv run lotus_pubchem_tree.py export -o ./output -v --format full
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
        parser.add_argument(
            "--format",
            "-f",
            choices=["pubchem", "full"],
            default="pubchem",
            help="Output format: 'pubchem' (default, compact name-as-key) or 'full' (detailed with metadata)",
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
                reference_doi=process_reference_doi(data.reference_doi),
                reference_pmid=process_reference_pmid(data.reference_pmid),
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
                    compound_label_df["compound"],
                    compound_label_df["compound_label"],
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

            # Build reference map for compound-taxon pairs
            if args.verbose:
                print("  Building reference map...", file=sys.stderr)
            reference_doi_df = data.reference_doi.collect()
            reference_pmid_df = data.reference_pmid.collect()
            reference_map = build_reference_map(reference_doi_df, reference_pmid_df)
            if args.verbose:
                print(
                    f"  Compounds with references: {len(reference_map):,}",
                    file=sys.stderr,
                )

            biological_tree = build_biological_tree(
                compound_taxon_df,
                taxon_ncbi_df,
                taxon_parent_df,
                taxon_name_df,
                compounds_with_taxa,
                descriptor_map,
                label_map,
                reference_map,
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

            # Build SMILES to InChIKey mapping for NPClassifier
            if args.verbose:
                print("\nBuilding SMILES to InChIKey mapping...", file=sys.stderr)
            smiles_to_inchikey, smiles_to_qid = build_smiles_to_inchikey_map(
                compound_taxon_df,
                smiles_map,
                inchikey_map,
            )

            # Build NPClassifier tree
            if args.verbose:
                print("\nFetching NPClassifier cache...", file=sys.stderr)
            npclassifier_df = fetch_npclassifier_cache()

            npclassifier_tree = []
            if len(npclassifier_df) > 0:
                if args.verbose:
                    print("Building NPClassifier chemical tree...", file=sys.stderr)
                npclassifier_tree = build_npclassifier_tree(
                    npclassifier_df,
                    smiles_to_inchikey,
                    smiles_to_qid,
                )
                if args.verbose:
                    npc_nodes = count_tree_nodes(npclassifier_tree)
                    print(f"  Root nodes: {len(npclassifier_tree)}", file=sys.stderr)
                    print(f"  Total nodes: {npc_nodes:,}", file=sys.stderr)
            else:
                if args.verbose:
                    print("  Could not fetch NPClassifier cache", file=sys.stderr)

            date_str = datetime.now().strftime("%Y%m%d")
            generated_at = datetime.now().isoformat()

            def build_cli_tree_output(
                tree_type: str,
                tree: list[dict],
                source: str = "wikidata",
            ) -> dict:
                """Build complete output with rich metadata for CLI."""
                is_biological = tree_type == "biological"
                is_npclassifier = source == "npclassifier"

                if is_npclassifier:
                    overview = "This JSON contains a hierarchical tree of chemical compounds classified using NPClassifier (pathway → superclass → class)."
                    description = "NPClassifier-based hierarchical classification of natural products"
                    notes = [
                        "NPClassifier provides a comprehensive classification for natural products.",
                        "Hierarchy: pathway → superclass → class → InChIKey",
                        "Multiple classifications are possible for a single compound.",
                        "See https://npclassifier.gnps2.org/ for more information.",
                    ]
                elif is_biological:
                    overview = "This JSON contains a hierarchical tree of biological taxa with their associated natural product compounds."
                    description = "Hierarchical taxonomy of biological organisms with associated natural product compounds"
                    notes = [
                        "Descriptor values are strings when single, arrays when multiple values exist (very rare).",
                        "All nodes are sorted alphabetically by Name.",
                        "Only nodes with InChIKey associations (directly or via descendants) are included.",
                        "Data queried from Wikidata via QLever SPARQL endpoint.",
                    ]
                else:
                    overview = "This JSON contains a hierarchical tree of chemical compound classes with structural descriptors based on Wikidata P279 relationships."
                    description = "Wikidata-based hierarchical classification of chemical compounds"
                    notes = [
                        "Note: Wikidata P279 (subclass of) relationships are sparse for natural products.",
                        "Consider using the NPClassifier-based tree (lotus_chemical_tree.json) for better coverage.",
                        "Descriptor values are strings when single, arrays when multiple values exist (very rare).",
                        "All nodes are sorted alphabetically by Name.",
                    ]

                return {
                    "_documentation": {
                        "overview": overview,
                        "structure": {
                            "tree": "Array of root nodes. Each node is an object with Name, Identifiers, and optional Children.",
                            "node_fields": {
                                "Name": "Human-readable name (taxon name or compound label)",
                                "Identifiers": "External database identifiers (Wikidata QID, NCBI TaxID for taxa, NPClassifier levels)",
                                "Compounds": "(Biological tree only) Array of compounds found in this taxon",
                                "Descriptors": "Chemical structure representations (InChIKey, SMILES, etc.)",
                                "References": "(Biological tree compounds only) Literature references for compound-taxon association",
                                "Children": "Array of child nodes (same structure, recursive)",
                            },
                            "descriptors_fields": {
                                "InChIKey": "IUPAC International Chemical Identifier Key",
                                "SMILES": "Simplified Molecular Input Line Entry System (isomeric preferred over canonical)",
                                "SMARTS": "SMILES Arbitrary Target Specification (substructure patterns)",
                                "CXSMILES": "ChemAxon Extended SMILES",
                            },
                            "reference_fields": {
                                "QID": "QID of the reference article in Wikidata",
                                "DOI": "Digital Object Identifier of the reference",
                                "PMID": "PubMed ID of the reference",
                            },
                        },
                        "notes": notes,
                    },
                    "metadata": {
                        "name": f"LOTUS {tree_type.title()} Tree"
                        + (
                            " (NPClassifier)"
                            if is_npclassifier
                            else (" (Wikidata)" if not is_biological else "")
                        ),
                        "description": description,
                        "version": CONFIG["app_version"],
                        "generated": generated_at,
                        "generator": CONFIG["app_name"],
                        "source": {
                            "name": METADATA["project"],
                            "url": METADATA["project_url"],
                            "wikidata_item": METADATA["wikidata_item"],
                            "endpoint": CONFIG["qlever_endpoint"]
                            if not is_npclassifier
                            else CONFIG["npclassifier_cache_url"],
                            "classification": "NPClassifier"
                            if is_npclassifier
                            else "Wikidata",
                        },
                        "license": {
                            "data": METADATA["license_data"],
                            "code": METADATA["license_code"],
                        },
                        "constraints": {
                            "root": "Biota (Q2382443)"
                            if is_biological
                            else (
                                "NPClassifier pathways"
                                if is_npclassifier
                                else "chemical compound (Q11173)"
                            ),
                        },
                        "statistics": {
                            "root_nodes": len(tree),
                            "total_nodes": count_tree_nodes(tree),
                        },
                    },
                    "tree": tree,
                }

            biological_output = build_cli_tree_output("biological", biological_tree)
            chemical_wikidata_output = build_cli_tree_output(
                "chemical",
                chemical_tree,
                source="wikidata",
            )
            npclassifier_output = (
                build_cli_tree_output(
                    "chemical",
                    npclassifier_tree,
                    source="npclassifier",
                )
                if npclassifier_tree
                else None
            )

            # Choose format based on --format option
            if args.format == "full":
                # Full format: detailed with metadata
                biological_final = biological_output
                chemical_wikidata_final = chemical_wikidata_output
                npclassifier_final = npclassifier_output

                biological_path = output_dir / f"{date_str}_lotus_biological_tree.json"
                # NPClassifier is the main chemical tree
                chemical_path = output_dir / f"{date_str}_lotus_chemical_tree.json"
                chemical_wikidata_path = (
                    output_dir / f"{date_str}_lotus_chemical_tree_wikidata.json"
                )
            else:
                # PubChem format (default): compact, name-as-key structure
                if args.verbose:
                    print("\nConverting to PubChem format...", file=sys.stderr)
                biological_final = tree_to_pubchem_format(biological_tree, "biological")
                chemical_wikidata_final = tree_to_pubchem_format(
                    chemical_tree,
                    "chemical",
                )
                npclassifier_final = (
                    npclassifier_tree_to_pubchem(npclassifier_tree)
                    if npclassifier_tree
                    else None
                )

                biological_path = output_dir / f"{date_str}_lotus_biological_tree.json"
                # NPClassifier is the main chemical tree
                chemical_path = output_dir / f"{date_str}_lotus_chemical_tree.json"
                chemical_wikidata_path = (
                    output_dir / f"{date_str}_lotus_chemical_tree_wikidata.json"
                )

            if args.verbose:
                print("\nWriting output files...", file=sys.stderr)

            biological_path.write_text(json.dumps(biological_final, indent=2))
            if args.verbose:
                print(f"  ✓ {biological_path}", file=sys.stderr)
            else:
                print(biological_path)

            # Write Wikidata chemical tree
            chemical_wikidata_path.write_text(
                json.dumps(chemical_wikidata_final, indent=2),
            )
            if args.verbose:
                print(f"  ✓ {chemical_wikidata_path}", file=sys.stderr)
            else:
                print(chemical_wikidata_path)

            # Write NPClassifier chemical tree as the main chemical tree
            if npclassifier_final:
                chemical_path.write_text(json.dumps(npclassifier_final, indent=2))
                if args.verbose:
                    print(
                        f"  ✓ {chemical_path} (NPClassifier)",
                        file=sys.stderr,
                    )
                else:
                    print(chemical_path)
            else:
                if args.verbose:
                    print(
                        "  NPClassifier tree not available, main chemical tree not written",
                        file=sys.stderr,
                    )

            if args.verbose:
                print("\nDone!", file=sys.stderr)

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
