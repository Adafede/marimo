# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "polars==1.39.3",
#     "rdkit==2026.3.1",
# ]
# [tool.marimo.runtime]
# output_max_bytes = 1_073_741_824
# ///

"""
LOTUS Data Exporter

Generates CSV/TSV files matching the Zenodo LOTUS data exports:
- https://doi.org/10.5281/zenodo.5794106 (frozen.csv + frozen_metadata.csv)
- https://doi.org/10.5281/zenodo.6378223 (organism_metadata.tsv, reference_metadata.tsv, structure_metadata.tsv)

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
    uv run https://adafede.github.io/marimo/apps/lotus_exporter.py export -o ./output -v

    # Remote GUI (interactive)
    uvx marimo run https://adafede.github.io/marimo/apps/lotus_exporter.py

    # Local GUI
    marimo run lotus_exporter.py

    # Local CLI export
    python lotus_exporter.py export -o ./output -v
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="LOTUS Data Exporter")

with app.setup:
    import marimo as mo
    import polars as pl
    import io
    import sys
    import gzip
    from pathlib import Path
    from dataclasses import dataclass
    from datetime import datetime
    from typing import cast

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
        "app_name": "LOTUS Data Exporter",
        "qlever_endpoint": "https://qlever.dev/api/wikidata",
        "pubchem_endpoint": "https://qlever.dev/api/pubchem",
        "npclassifier_cache_url": "https://adafede.github.io/marimo/apps/public/npclassifier/npclassifier_cache.csv",
        "classyfire_cache_url": "https://adafede.github.io/marimo/apps/public/classyfire/classyfire_cache.csv",
        "ott_cache_url": "https://adafede.github.io/marimo/apps/public/ott/ott.tsv",
    }

    METADATA = {
        "project": "LOTUS",
        "project_url": "https://lotus.nprod.net/",
        "wikidata_item": "Q104225190",
        "source": "Wikidata via QLever SPARQL endpoint",
        "license_data": "CC0 1.0 Universal",
        "license_code": "AGPL-3.0",
        "zenodo_frozen": "https://zenodo.org/records/7534071",
    }

    # ========================================================================
    # SPARQL QUERIES
    # ========================================================================

    # Core triplet: compound-taxon-reference
    QUERY_COMPOUND_TAXON_REFERENCE = """
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?compound ?compound_inchikey ?taxon ?reference WHERE {
      ?compound wdt:P235 ?compound_inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
    }
    """

    # Structure metadata - restricted to compounds with InChIKey + taxon
    QUERY_COMPOUND_SMILES_CAN = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?compound ?compound_smiles_can WHERE {
      ?compound wdt:P235 ?inchikey ;
               wdt:P703 ?taxon ;
               wdt:P233 ?compound_smiles_can .
    }
    """

    QUERY_COMPOUND_SMILES_ISO = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?compound ?compound_smiles_iso WHERE {
      ?compound wdt:P235 ?inchikey ;
               wdt:P703 ?taxon ;
               wdt:P2017 ?compound_smiles_iso .
    }
    """

    QUERY_COMPOUND_CID = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?compound ?compound_cid WHERE {
      ?compound wdt:P235 ?inchikey ;
               wdt:P703 ?taxon ;
               wdt:P662 ?compound_cid .
    }
    """

    # Taxon metadata
    QUERY_TAXON_NAME = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?taxon ?taxon_name WHERE {
      ?taxon wdt:P225 ?taxon_name .
    }
    """

    QUERY_TAXON_NCBI = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?taxon ?taxon_ncbi WHERE {
      ?taxon wdt:P685 ?taxon_ncbi .
    }
    """

    QUERY_TAXON_OTT = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?taxon ?taxon_ott WHERE {
      ?taxon wdt:P9157 ?taxon_ott .
    }
    """

    QUERY_TAXON_GBIF = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?taxon ?taxon_gbif WHERE {
      ?taxon wdt:P846 ?taxon_gbif .
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

    QUERY_TAXON_RANK = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?taxon ?taxon_rank ?taxon_rank_label WHERE {
      ?taxon wdt:P105 ?taxon_rank .
      ?taxon_rank rdfs:label ?taxon_rank_label .
      FILTER(LANG(?taxon_rank_label) = "en")
    }
    """

    # Reference metadata - restricted to references from compound-taxon statements
    QUERY_REFERENCE_DOI = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?reference ?doi WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P356 ?doi .
    }
    """

    QUERY_REFERENCE_PMID = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?reference ?pmid WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P698 ?pmid .
    }
    """

    QUERY_REFERENCE_PMCID = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?reference ?pmcid WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P932 ?pmcid .
    }
    """

    QUERY_REFERENCE_TITLE = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?reference ?title WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P1476 ?title .
    }
    """

    QUERY_REFERENCE_DATE = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?reference ?date WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P577 ?date .
    }
    """

    QUERY_REFERENCE_JOURNAL = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    SELECT DISTINCT ?reference ?journal ?journal_label WHERE {
      ?compound wdt:P235 ?inchikey .
      ?compound p:P703 ?statement .
      ?statement ps:P703 ?taxon .
      ?statement prov:wasDerivedFrom/pr:P248 ?reference .
      ?reference wdt:P1433 ?journal .
      ?journal wdt:P1476 ?journal_label .
    }
    """

    # PubChem SPARQL query template for fetching compound data (batched with VALUES)
    # Fetches: CID, InChIKey, mass, SMILES (isomeric), connectivity SMILES (2D), names, stereo counts
    # The {cid_values} placeholder is replaced with VALUES clause for each batch
    QUERY_PUBCHEM_COMPOUNDS_TEMPLATE = """
    PREFIX compound: <http://rdf.ncbi.nlm.nih.gov/pubchem/compound/>
    PREFIX vocab: <http://rdf.ncbi.nlm.nih.gov/pubchem/vocabulary#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?cid ?inchikey ?mono_isotopic_weight ?smiles ?connectivity_smiles ?common_name ?iupac_name ?iupac_inchi ?molecular_formula ?xlogp3 ?defined_atom_stereo_count ?defined_bond_stereo_count ?undefined_atom_stereo_count ?undefined_bond_stereo_count WHERE {{
      VALUES ?cid {{ {cid_values} }}
      ?cpd dcterms:identifier ?cid ;
           vocab:inchikey ?inchikey .
      OPTIONAL {{ ?cpd vocab:mono_isotopic_weight ?mono_isotopic_weight . }}
      OPTIONAL {{ ?cpd vocab:smiles ?smiles . }}
      OPTIONAL {{ ?cpd vocab:connectivity_smiles ?connectivity_smiles . }}
      OPTIONAL {{ ?cpd skos:prefLabel ?common_name . }}
      OPTIONAL {{ ?cpd vocab:preferred_iupac_name ?iupac_name . }}
      OPTIONAL {{ ?cpd vocab:iupac_inchi ?iupac_inchi . }}
      OPTIONAL {{ ?cpd vocab:molecular_formula ?molecular_formula . }}
      OPTIONAL {{ ?cpd vocab:xlogp3 ?xlogp3 . }}
      OPTIONAL {{ ?cpd vocab:defined_atom_stereo_count ?defined_atom_stereo_count . }}
      OPTIONAL {{ ?cpd vocab:defined_bond_stereo_count ?defined_bond_stereo_count . }}
      OPTIONAL {{ ?cpd vocab:undefined_atom_stereo_count ?undefined_atom_stereo_count . }}
      OPTIONAL {{ ?cpd vocab:undefined_bond_stereo_count ?undefined_bond_stereo_count . }}
    }}
    """

    # PubChem query by InChIKey (for compounds without CID in Wikidata)
    QUERY_PUBCHEM_BY_INCHIKEY_TEMPLATE = """
    PREFIX compound: <http://rdf.ncbi.nlm.nih.gov/pubchem/compound/>
    PREFIX vocab: <http://rdf.ncbi.nlm.nih.gov/pubchem/vocabulary#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?cid ?inchikey ?mono_isotopic_weight ?smiles ?connectivity_smiles ?common_name ?iupac_name ?iupac_inchi ?molecular_formula ?xlogp3 ?defined_atom_stereo_count ?defined_bond_stereo_count ?undefined_atom_stereo_count ?undefined_bond_stereo_count WHERE {{
      VALUES ?inchikey {{ {inchikey_values} }}
      ?cpd vocab:inchikey ?inchikey ;
           dcterms:identifier ?cid .
      OPTIONAL {{ ?cpd vocab:mono_isotopic_weight ?mono_isotopic_weight . }}
      OPTIONAL {{ ?cpd vocab:smiles ?smiles . }}
      OPTIONAL {{ ?cpd vocab:connectivity_smiles ?connectivity_smiles . }}
      OPTIONAL {{ ?cpd skos:prefLabel ?common_name . }}
      OPTIONAL {{ ?cpd vocab:preferred_iupac_name ?iupac_name . }}
      OPTIONAL {{ ?cpd vocab:iupac_inchi ?iupac_inchi . }}
      OPTIONAL {{ ?cpd vocab:molecular_formula ?molecular_formula . }}
      OPTIONAL {{ ?cpd vocab:xlogp3 ?xlogp3 . }}
      OPTIONAL {{ ?cpd vocab:defined_atom_stereo_count ?defined_atom_stereo_count . }}
      OPTIONAL {{ ?cpd vocab:defined_bond_stereo_count ?defined_bond_stereo_count . }}
      OPTIONAL {{ ?cpd vocab:undefined_atom_stereo_count ?undefined_atom_stereo_count . }}
      OPTIONAL {{ ?cpd vocab:undefined_bond_stereo_count ?undefined_bond_stereo_count . }}
    }}
    """

    # Batch size for PubChem queries (number of CIDs per query)
    PUBCHEM_BATCH_SIZE = 5000

    # ========================================================================
    # PREVIOUS VERSION LOADING & CHANGE TRACKING
    # ========================================================================

    # Zenodo record IDs (latest versions)
    ZENODO_FROZEN_RECORD_ID = (
        "7534071"  # frozen.csv - https://zenodo.org/records/7534071
    )

    CRITICAL_STRUCTURE_COLS = [
        "structure_inchi",
        "structure_smiles",
        "structure_molecular_formula",
        "structure_exact_mass",
        "structure_smiles_2D",
        "structure_stereocenters_total",
        "structure_stereocenters_unspecified",
        # Not really a structure, but easier
        "organism_name",
    ]

    def drop_missing_structure_rows(
        df: pl.DataFrame,
        verbose: bool = False,
    ) -> pl.DataFrame:
        """Drop rows where any critical structure field is null.
                                These rows have no usable chemical data and should not be published.

        Parameters
        ----------
        df : pl.DataFrame
            Df.
        verbose : bool
            False. Default is False.

        Returns
        -------
        pl.DataFrame
            Input rows with missing critical structure fields removed.
        """
        before = len(df)
        mask = pl.all_horizontal(
            [pl.col(c).is_not_null() for c in CRITICAL_STRUCTURE_COLS],
        )
        df = df.filter(mask)
        dropped = before - len(df)
        if dropped > 0:
            print(
                f"  Dropped {dropped:,} rows with missing critical structure fields",
                file=sys.stderr,
            )
        return df

    def fetch_latest_zenodo_frozen() -> pl.DataFrame:
        """Fetch the latest frozen.csv from Zenodo to inherit manual_validation.
                                Uses the Zenodo API to find the latest version and download frozen.csv.

        Returns
        -------
        pl.DataFrame
            Previously published ``frozen.csv`` records loaded from Zenodo.
        """
        import urllib.request
        import json

        try:
            # Get the latest version info from Zenodo API
            api_url = f"https://zenodo.org/api/records/{ZENODO_FROZEN_RECORD_ID}"
            with urllib.request.urlopen(api_url, timeout=30) as resp:
                record_data = json.loads(resp.read().decode("utf-8"))

            # Find the frozen.csv file in the files list
            files = record_data.get("files", [])
            frozen_file = None
            for f in files:
                key = f.get("key", "")
                # Match files like "230106_frozen.csv.gz" or "frozen.csv" (exclude frozen_metadata)
                if "frozen.csv" in key and "metadata" not in key:
                    frozen_file = f
                    break

            if not frozen_file:
                # Try any frozen file as fallback
                for f in files:
                    key = f.get("key", "").lower()
                    if "frozen" in key and "metadata" not in key:
                        frozen_file = f
                        break

            if not frozen_file:
                print(
                    "Warning: Could not find frozen.csv in Zenodo record",
                    file=sys.stderr,
                )
                return pl.DataFrame()

            # Download the file
            download_url = frozen_file.get("links", {}).get("self")
            if not download_url:
                # Construct download URL from key
                record_id = record_data.get("id", ZENODO_FROZEN_RECORD_ID)
                download_url = f"https://zenodo.org/records/{record_id}/files/{frozen_file['key']}?download=1"

            print(f"  Downloading {frozen_file['key']} from Zenodo...", file=sys.stderr)
            with urllib.request.urlopen(download_url, timeout=120) as resp:
                data = resp.read()

            # Handle gzipped files
            if frozen_file.get("key", "").endswith(".gz") or data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
                # Handle double-gzipped files
                if data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)

            return pl.read_csv(
                io.BytesIO(data),
                infer_schema_length=0,  # All strings
            )
        except Exception as e:
            print(
                f"Warning: Could not fetch previous frozen from Zenodo: {e}",
                file=sys.stderr,
            )
            return pl.DataFrame()

    def inherit_manual_validation(
        new_df: pl.DataFrame,
        old_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Inherit manual_validation from previous version.
                                Matches on structure_inchikey + organism_wikidata + reference_wikidata.

        Parameters
        ----------
        new_df : pl.DataFrame
            New df.
        old_df : pl.DataFrame
            Old df.

        Returns
        -------
        pl.DataFrame
            New dataset with ``manual_validation`` values inherited where keys match prior releases.
        """
        if len(old_df) == 0:
            return new_df

        # Get validation data from old
        validation_data = old_df.filter(
            pl.col("manual_validation") != "NA",
        ).select(
            [
                "structure_inchikey",
                "organism_wikidata",
                "reference_wikidata",
                pl.col("manual_validation").alias("old_validation"),
            ],
        )

        if len(validation_data) == 0:
            return new_df

        # Join to inherit validation
        result = (
            new_df.join(
                validation_data,
                on=["structure_inchikey", "organism_wikidata", "reference_wikidata"],
                how="left",
            )
            .with_columns(
                pl.coalesce(["old_validation", "manual_validation"]).alias(
                    "manual_validation",
                ),
            )
            .drop("old_validation")
        )

        return normalize_blank_strings(result)

    def compute_changes(
        new_df: pl.DataFrame,
        old_df: pl.DataFrame,
        key_cols: list[str],
        name: str,
    ) -> dict:
        """Compute changes between new and old versions.
                                Returns dict with added, removed counts and samples.

        Parameters
        ----------
        new_df : pl.DataFrame
            New df.
        old_df : pl.DataFrame
            Old df.
        key_cols : list[str]
            Key cols.
        name : str
            Name.

        Returns
        -------
        dict
            Summary counts for added, removed, unchanged, and total records.
        """
        if len(old_df) == 0:
            return {
                "name": name,
                "added": len(new_df),
                "removed": 0,
                "unchanged": 0,
                "total_new": len(new_df),
                "total_old": 0,
            }

        # Check if key columns exist in old DataFrame
        missing_cols = [col for col in key_cols if col not in old_df.columns]
        if missing_cols:
            # Old format has different schema, can't compare
            return {
                "name": name,
                "added": len(new_df),
                "removed": 0,
                "unchanged": 0,
                "total_new": len(new_df),
                "total_old": len(old_df),
                "note": f"Schema mismatch: missing columns {missing_cols}",
            }

        # Create composite keys
        new_keys = set(
            new_df.select(key_cols).unique().iter_rows(),
        )
        old_keys = set(
            old_df.select(key_cols).unique().iter_rows(),
        )

        added = new_keys - old_keys
        removed = old_keys - new_keys
        unchanged = new_keys & old_keys

        return {
            "name": name,
            "added": len(added),
            "removed": len(removed),
            "unchanged": len(unchanged),
            "total_new": len(new_keys),
            "total_old": len(old_keys),
        }

    def write_changes_report(
        changes: list[dict],
        output_path: str,
    ):
        """Write changes report to a text file.

        Parameters
        ----------
        changes : list[dict]
            Changes.
        output_path : str
            Output path.
        """

        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("LOTUS Data Export - Changes Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

            for change in changes:
                f.write(f"## {change['name']}\n")
                f.write(f"  Previous version: {change['total_old']:,} entries\n")
                f.write(f"  Current version:  {change['total_new']:,} entries\n")
                f.write(f"  Added:            +{change['added']:,}\n")
                f.write(f"  Removed:          -{change['removed']:,}\n")
                f.write(f"  Unchanged:        {change['unchanged']:,}\n")
                f.write("\n")

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def extract_qid(url: str) -> str:
        """Extract QID from Wikidata URL.

        Parameters
        ----------
        url : str
            Url.

        Returns
        -------
        str
            URL (or original value).
        """
        if url and url.startswith(WIKIDATA_ENTITY_PREFIX):
            return url.replace(WIKIDATA_ENTITY_PREFIX, "")
        return url

    def execute_query(
        query: str,
        endpoint: str,
    ) -> pl.LazyFrame:
        """Execute SPARQL query and return LazyFrame.

        Parameters
        ----------
        query : str
            Query.
        endpoint : str
            Endpoint.

        Returns
        -------
        pl.LazyFrame
            Polars frame.
        """
        csv_bytes = execute_with_retry(query, endpoint, timeout=600)
        if not csv_bytes or len(csv_bytes) < 10:
            return pl.LazyFrame()

        return pl.scan_csv(
            io.BytesIO(csv_bytes),
            rechunk=False,
            infer_schema_length=0,  # Don't infer, treat all as strings
        )

    # ========================================================================
    # DATA STRUCTURES
    # ========================================================================

    @dataclass
    class LOTUSExportData:
        """Container for all fetched LOTUS data for export."""

        # Core triplets
        compound_taxon_reference: pl.LazyFrame

        # Structure metadata
        compound_smiles_can: pl.LazyFrame
        compound_smiles_iso: pl.LazyFrame
        compound_cid: pl.LazyFrame

        # Taxon metadata
        taxon_name: pl.LazyFrame
        taxon_ncbi: pl.LazyFrame
        taxon_ott: pl.LazyFrame
        taxon_gbif: pl.LazyFrame
        taxon_parent: pl.LazyFrame
        taxon_rank: pl.LazyFrame

        # Reference metadata
        reference_doi: pl.LazyFrame
        reference_pmid: pl.LazyFrame
        reference_pmcid: pl.LazyFrame
        reference_title: pl.LazyFrame
        reference_date: pl.LazyFrame
        reference_journal: pl.LazyFrame

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    def fetch_all_data(endpoint: str, progress_callback=None) -> LOTUSExportData:
        """Fetch all required data from Wikidata.

        Parameters
        ----------
        endpoint : str
            Endpoint.
        progress_callback : Any
            None. Default is None.

        Returns
        -------
        LOTUSExportData
            Bundle containing all fetched datasets required for export.
        """
        queries = [
            (
                "compound_taxon_reference",
                QUERY_COMPOUND_TAXON_REFERENCE,
                "Fetching compound-taxon-reference triplets...",
            ),
            (
                "compound_smiles_can",
                QUERY_COMPOUND_SMILES_CAN,
                "Fetching canonical SMILES...",
            ),
            (
                "compound_smiles_iso",
                QUERY_COMPOUND_SMILES_ISO,
                "Fetching isomeric SMILES...",
            ),
            ("compound_cid", QUERY_COMPOUND_CID, "Fetching PubChem CIDs..."),
            ("taxon_name", QUERY_TAXON_NAME, "Fetching taxon names..."),
            ("taxon_ncbi", QUERY_TAXON_NCBI, "Fetching NCBI taxonomy IDs..."),
            ("taxon_ott", QUERY_TAXON_OTT, "Fetching OTT IDs..."),
            ("taxon_gbif", QUERY_TAXON_GBIF, "Fetching GBIF IDs..."),
            ("taxon_parent", QUERY_TAXON_PARENT, "Fetching taxon parents..."),
            ("taxon_rank", QUERY_TAXON_RANK, "Fetching taxon ranks..."),
            ("reference_doi", QUERY_REFERENCE_DOI, "Fetching DOIs..."),
            ("reference_pmid", QUERY_REFERENCE_PMID, "Fetching PMIDs..."),
            ("reference_pmcid", QUERY_REFERENCE_PMCID, "Fetching PMCIDs..."),
            ("reference_title", QUERY_REFERENCE_TITLE, "Fetching reference titles..."),
            ("reference_date", QUERY_REFERENCE_DATE, "Fetching publication dates..."),
            ("reference_journal", QUERY_REFERENCE_JOURNAL, "Fetching journals..."),
        ]

        results = {}
        for name, query, msg in queries:
            if progress_callback:
                progress_callback(msg)
            results[name] = execute_query(query, endpoint)

        return LOTUSExportData(**results)

    # ========================================================================
    # EXTERNAL CACHE FETCHING
    # ========================================================================

    def fetch_npclassifier_cache(url: str | None = None) -> pl.DataFrame:
        """Fetch NPClassifier cache CSV from remote URL.
                                Returns DataFrame with columns: smiles, pathway, superclass, class, isglycoside, error

        Parameters
        ----------
        url : str | None
            None. Default is None.

        Returns
        -------
        pl.DataFrame
            SMILES entry.
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

    def fetch_classyfire_cache(url: str | None = None) -> pl.DataFrame:
        """Fetch ClassyFire cache from remote URL or local file.
                                Expected columns: inchikey, chemontid, kingdom, superclass, class, direct_parent

        Parameters
        ----------
        url : str | None
            None. Default is None.

        Returns
        -------
        pl.DataFrame
            InChIKey.
        """

        def _empty_classyfire_df() -> pl.DataFrame:
            return pl.DataFrame(
                schema={
                    "inchikey": pl.Utf8,
                    "chemontid": pl.Utf8,
                    "kingdom": pl.Utf8,
                    "superclass": pl.Utf8,
                    "class": pl.Utf8,
                    "direct_parent": pl.Utf8,
                },
            )

        configured = url or CONFIG.get("classyfire_cache_url")
        candidates = [
            c for c in [configured, "apps/public/classyfire/classyfire_cache.csv"] if c
        ]

        for source in candidates:
            try:
                if source.startswith("http://") or source.startswith("https://"):
                    df = pl.read_csv(source)
                else:
                    if not Path(source).exists():
                        continue
                    df = pl.read_csv(source)

                if len(df) == 0:
                    continue

                cols = set(df.columns)
                mapped = df.select(
                    [
                        (
                            pl.col("inchikey")
                            if "inchikey" in cols
                            else pl.col("structure_inchikey")
                        )
                        .cast(pl.Utf8)
                        .alias("inchikey"),
                        (
                            pl.col("chemontid")
                            if "chemontid" in cols
                            else pl.col("structure_taxonomy_classyfire_chemontid")
                        )
                        .cast(pl.Utf8)
                        .alias("chemontid"),
                        (
                            pl.col("kingdom")
                            if "kingdom" in cols
                            else pl.col("structure_taxonomy_classyfire_01kingdom")
                        )
                        .cast(pl.Utf8)
                        .alias("kingdom"),
                        (
                            pl.col("superclass")
                            if "superclass" in cols
                            else pl.col("structure_taxonomy_classyfire_02superclass")
                        )
                        .cast(pl.Utf8)
                        .alias("superclass"),
                        (
                            pl.col("class")
                            if "class" in cols
                            else pl.col("structure_taxonomy_classyfire_03class")
                        )
                        .cast(pl.Utf8)
                        .alias("class"),
                        (
                            pl.col("direct_parent")
                            if "direct_parent" in cols
                            else pl.col("structure_taxonomy_classyfire_04directparent")
                        )
                        .cast(pl.Utf8)
                        .alias("direct_parent"),
                    ],
                ).unique(subset=["inchikey"])

                return normalize_blank_strings(mapped)
            except Exception as e:
                print(
                    f"Warning: Could not read ClassyFire cache from {source}: {e}",
                    file=sys.stderr,
                )

        return _empty_classyfire_df()

    def fetch_ott_taxonomy_cache(url: str | None = None) -> pl.DataFrame:
        """Fetch Open Tree of Life (OTT) taxonomy cache from remote URL or local file.
                                Expected columns: organism_name, organism_taxonomy_ottid, organism_taxonomy_01domain, etc.

                                Maps to: organism_name, ott_id, domain, kingdom, phylum, class, order, family, tribe, genus, species, varietas

        Parameters
        ----------
        url : str | None
            None. Default is None.

        Returns
        -------
        pl.DataFrame
            OTT taxonomy cache table mapped to the export taxonomy columns.
        """
        url = url or CONFIG.get("ott_cache_url")
        if not url:
            return pl.DataFrame(
                schema={
                    "organism_name": pl.Utf8,
                    "ott_id": pl.Utf8,
                    "domain": pl.Utf8,
                    "kingdom": pl.Utf8,
                    "phylum": pl.Utf8,
                    "class": pl.Utf8,
                    "order": pl.Utf8,
                    "family": pl.Utf8,
                    "tribe": pl.Utf8,
                    "genus": pl.Utf8,
                    "species": pl.Utf8,
                    "varietas": pl.Utf8,
                },
            )

        def _empty_ott_df() -> pl.DataFrame:
            return pl.DataFrame(
                schema={
                    "organism_name": pl.Utf8,
                    "ott_id": pl.Utf8,
                    "domain": pl.Utf8,
                    "kingdom": pl.Utf8,
                    "phylum": pl.Utf8,
                    "class": pl.Utf8,
                    "order": pl.Utf8,
                    "family": pl.Utf8,
                    "tribe": pl.Utf8,
                    "genus": pl.Utf8,
                    "species": pl.Utf8,
                    "varietas": pl.Utf8,
                },
            )

        configured = url or CONFIG.get("ott_cache_url")
        candidates = [c for c in [configured, "apps/public/ott/ott.tsv"] if c]

        for source in candidates:
            try:
                if source.startswith("http://") or source.startswith("https://"):
                    df = pl.read_csv(source, separator="\t")
                else:
                    if not Path(source).exists():
                        continue
                    df = pl.read_csv(source, separator="\t")

                if len(df) == 0:
                    continue

                cols = set(df.columns)
                mapped = df.select(
                    [
                        (
                            pl.col("organism_name")
                            if "organism_name" in cols
                            else pl.col("taxon_name")
                        )
                        .cast(pl.Utf8)
                        .alias("organism_name"),
                        (
                            pl.col("organism_taxonomy_ottid")
                            if "organism_taxonomy_ottid" in cols
                            else pl.col("ott_id")
                        )
                        .cast(pl.Utf8)
                        .alias("ott_id"),
                        (
                            pl.col("organism_taxonomy_01domain")
                            if "organism_taxonomy_01domain" in cols
                            else pl.col("domain")
                        )
                        .cast(pl.Utf8)
                        .alias("domain"),
                        (
                            pl.col("organism_taxonomy_02kingdom")
                            if "organism_taxonomy_02kingdom" in cols
                            else pl.col("kingdom")
                        )
                        .cast(pl.Utf8)
                        .alias("kingdom"),
                        (
                            pl.col("organism_taxonomy_03phylum")
                            if "organism_taxonomy_03phylum" in cols
                            else pl.col("phylum")
                        )
                        .cast(pl.Utf8)
                        .alias("phylum"),
                        (
                            pl.col("organism_taxonomy_04class")
                            if "organism_taxonomy_04class" in cols
                            else pl.col("class")
                        )
                        .cast(pl.Utf8)
                        .alias("class"),
                        (
                            pl.col("organism_taxonomy_05order")
                            if "organism_taxonomy_05order" in cols
                            else pl.col("order")
                        )
                        .cast(pl.Utf8)
                        .alias("order"),
                        (
                            pl.col("organism_taxonomy_06family")
                            if "organism_taxonomy_06family" in cols
                            else pl.col("family")
                        )
                        .cast(pl.Utf8)
                        .alias("family"),
                        (
                            pl.col("organism_taxonomy_07tribe")
                            if "organism_taxonomy_07tribe" in cols
                            else pl.col("tribe")
                        )
                        .cast(pl.Utf8)
                        .alias("tribe"),
                        (
                            pl.col("organism_taxonomy_08genus")
                            if "organism_taxonomy_08genus" in cols
                            else pl.col("genus")
                        )
                        .cast(pl.Utf8)
                        .alias("genus"),
                        (
                            pl.col("organism_taxonomy_09species")
                            if "organism_taxonomy_09species" in cols
                            else pl.col("species")
                        )
                        .cast(pl.Utf8)
                        .alias("species"),
                        (
                            pl.col("organism_taxonomy_10varietas")
                            if "organism_taxonomy_10varietas" in cols
                            else pl.col("varietas")
                        )
                        .cast(pl.Utf8)
                        .alias("varietas"),
                    ],
                )
                return normalize_blank_strings(mapped)
            except Exception as e:
                print(
                    f"Warning: Could not read OTT cache from {source}: {e}",
                    file=sys.stderr,
                )

        return _empty_ott_df()

    def compute_rdkit_properties(smiles_list: list[str]) -> pl.DataFrame:
        """Compute RDKit-derived structure properties from a list of SMILES.

                                Returns DataFrame with columns:
                                - smiles (input SMILES)
                                - smiles_2D (canonical SMILES without stereochemistry)
                                - molecular_formula
                                - exact_mass
                                - xlogp
                                - stereocenters_total
                                - stereocenters_unspecified
                                - inchi (computed InChI)
                                - inchikey (computed InChIKey)

                                Based on: https://github.com/lotusnprod/lotus-processor/blob/main/src/2_curating/2_editing/structure/3_processingAndEnriching/chemosanitizer_functions.py

        Parameters
        ----------
        smiles_list : list[str]
            Smiles list.

        Returns
        -------
        pl.DataFrame
            RDKit-derived structure properties for valid input SMILES.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, inchi
            from rdkit.Chem.rdCIPLabeler import AssignCIPLabels
        except ImportError:
            print(
                "Warning: RDKit not available, skipping structure property computation",
                file=sys.stderr,
            )
            return pl.DataFrame(
                schema={
                    "smiles": pl.Utf8,
                    "smiles_2D": pl.Utf8,
                    "molecular_formula": pl.Utf8,
                    "exact_mass": pl.Utf8,
                    "xlogp": pl.Utf8,
                    "stereocenters_total": pl.Utf8,
                    "stereocenters_unspecified": pl.Utf8,
                    "inchi": pl.Utf8,
                    "inchikey": pl.Utf8,
                },
            )

        results = []
        for smiles in smiles_list:
            if not smiles:
                continue
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Compute properties
                molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
                exact_mass = Descriptors.ExactMolWt(mol)
                xlogp = Descriptors.MolLogP(mol)

                # Compute InChI and InChIKey
                try:
                    computed_inchi = inchi.MolToInchi(mol)
                    computed_inchikey = (
                        inchi.MolToInchiKey(mol) if computed_inchi else None
                    )
                except Exception:
                    computed_inchi = None
                    computed_inchikey = None

                # Count stereocenters
                try:
                    AssignCIPLabels(mol)
                    chiral_centers = Chem.FindMolChiralCenters(
                        mol,
                        includeUnassigned=True,
                    )
                    stereocenters_total = len(chiral_centers)
                    stereocenters_unspecified = sum(
                        1 for _, label in chiral_centers if label == "?"
                    )
                except Exception:
                    stereocenters_total = 0
                    stereocenters_unspecified = 0

                # Generate 2D SMILES (canonical, no stereochemistry)
                mol_2d = Chem.MolFromSmiles(smiles)
                if mol_2d:
                    Chem.RemoveStereochemistry(mol_2d)
                    smiles_2d = Chem.MolToSmiles(mol_2d)
                else:
                    smiles_2d = None

                results.append(
                    {
                        "smiles": smiles,
                        "smiles_2D": smiles_2d,
                        "molecular_formula": molecular_formula,
                        "exact_mass": str(round(exact_mass, 6)) if exact_mass else None,
                        "xlogp": str(round(xlogp, 2)) if xlogp else None,
                        "stereocenters_total": str(stereocenters_total),
                        "stereocenters_unspecified": str(stereocenters_unspecified),
                        "inchi": computed_inchi,
                        "inchikey": computed_inchikey,
                    },
                )
            except Exception:
                continue

        if not results:
            return pl.DataFrame(
                schema={
                    "smiles": pl.Utf8,
                    "smiles_2D": pl.Utf8,
                    "molecular_formula": pl.Utf8,
                    "exact_mass": pl.Utf8,
                    "xlogp": pl.Utf8,
                    "stereocenters_total": pl.Utf8,
                    "stereocenters_unspecified": pl.Utf8,
                    "inchi": pl.Utf8,
                    "inchikey": pl.Utf8,
                },
            )

        return pl.DataFrame(results)

    def fetch_pubchem_compound_data(
        cid_list: list[str],
        progress_callback=None,
    ) -> pl.DataFrame:
        """Fetch comprehensive compound data from PubChem via SPARQL.

                                Uses batched queries with VALUES clause for efficient filtering server-side.
                                This avoids downloading the entire PubChem database (120M+ records).

        Parameters
        ----------
        cid_list : list[str]
            Cid list.
        progress_callback : Any
            None. Default is None.

        Returns
        -------
        pl.DataFrame
            CIDs.
        """
        schema = {
            "cid": pl.Utf8,
            "pubchem_inchikey": pl.Utf8,
            "pubchem_mass": pl.Utf8,
            "pubchem_smiles": pl.Utf8,
            "pubchem_smiles_2d": pl.Utf8,
            "common_name": pl.Utf8,
            "iupac_name": pl.Utf8,
            "pubchem_inchi": pl.Utf8,
            "pubchem_formula": pl.Utf8,
            "pubchem_xlogp3": pl.Utf8,
            "pubchem_defined_atom_stereo_count": pl.Utf8,
            "pubchem_defined_bond_stereo_count": pl.Utf8,
            "pubchem_undefined_atom_stereo_count": pl.Utf8,
            "pubchem_undefined_bond_stereo_count": pl.Utf8,
        }

        if not cid_list:
            return pl.DataFrame(schema=schema)

        endpoint = CONFIG["pubchem_endpoint"]
        unique_cids = sorted({c for c in cid_list if c})

        # Process in batches to avoid query size limits
        batch_size = PUBCHEM_BATCH_SIZE
        all_results: list[pl.DataFrame] = []
        total_batches = (len(unique_cids) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(unique_cids))
            batch_cids = unique_cids[start_idx:end_idx]

            if progress_callback:
                progress_callback(batch_idx + 1, total_batches)

            # Build VALUES clause with quoted CIDs
            cid_values = " ".join(f'"{cid}"' for cid in batch_cids)
            query = QUERY_PUBCHEM_COMPOUNDS_TEMPLATE.format(cid_values=cid_values)

            csv_bytes = execute_with_retry(query, endpoint, timeout=300)
            if not csv_bytes or len(csv_bytes) < 10:
                continue  # Skip failed batches

            try:
                batch_df = pl.read_csv(
                    io.BytesIO(csv_bytes),
                    infer_schema_length=0,
                )
                if len(batch_df) > 0:
                    # Rename columns to match expected schema
                    batch_df = batch_df.select(
                        [
                            pl.col("cid"),
                            pl.col("inchikey").alias("pubchem_inchikey"),
                            pl.col("mono_isotopic_weight").alias("pubchem_mass"),
                            pl.col("smiles").alias("pubchem_smiles"),
                            pl.col("connectivity_smiles").alias("pubchem_smiles_2d"),
                            pl.col("common_name"),
                            pl.col("iupac_name"),
                            pl.col("iupac_inchi").alias("pubchem_inchi"),
                            pl.col("molecular_formula").alias("pubchem_formula"),
                            pl.col("xlogp3").alias("pubchem_xlogp3"),
                            pl.col("defined_atom_stereo_count").alias(
                                "pubchem_defined_atom_stereo_count",
                            ),
                            pl.col("defined_bond_stereo_count").alias(
                                "pubchem_defined_bond_stereo_count",
                            ),
                            pl.col("undefined_atom_stereo_count").alias(
                                "pubchem_undefined_atom_stereo_count",
                            ),
                            pl.col("undefined_bond_stereo_count").alias(
                                "pubchem_undefined_bond_stereo_count",
                            ),
                        ],
                    )
                    all_results.append(batch_df)
            except Exception as e:
                print(
                    f"Warning: Could not parse PubChem batch {batch_idx + 1}: {e}",
                    file=sys.stderr,
                )
                continue

        if not all_results:
            print("Warning: Could not fetch any PubChem compound data", file=sys.stderr)
            return pl.DataFrame(schema=schema)

        # Combine all batches and deduplicate
        result_df = pl.concat(all_results).unique(subset=["cid"])
        return result_df

    def fetch_pubchem_by_inchikey(
        inchikey_list: list[str],
        progress_callback=None,
    ) -> pl.DataFrame:
        """Fetch PubChem compound data by InChIKey for compounds without CIDs.

                                Uses batched queries with VALUES clause for efficient filtering server-side.

        Parameters
        ----------
        inchikey_list : list[str]
            Inchikey list.
        progress_callback : Any
            None. Default is None.

        Returns
        -------
        pl.DataFrame
            InChIKeys.
        """
        schema = {
            "cid": pl.Utf8,
            "pubchem_inchikey": pl.Utf8,
            "pubchem_mass": pl.Utf8,
            "pubchem_smiles": pl.Utf8,
            "pubchem_smiles_2d": pl.Utf8,
            "common_name": pl.Utf8,
            "iupac_name": pl.Utf8,
            "pubchem_inchi": pl.Utf8,
            "pubchem_formula": pl.Utf8,
            "pubchem_xlogp3": pl.Utf8,
            "pubchem_defined_atom_stereo_count": pl.Utf8,
            "pubchem_defined_bond_stereo_count": pl.Utf8,
            "pubchem_undefined_atom_stereo_count": pl.Utf8,
            "pubchem_undefined_bond_stereo_count": pl.Utf8,
        }

        if not inchikey_list:
            return pl.DataFrame(schema=schema)

        endpoint = CONFIG["pubchem_endpoint"]
        unique_inchikeys = sorted({ik for ik in inchikey_list if ik})

        batch_size = PUBCHEM_BATCH_SIZE
        all_results: list[pl.DataFrame] = []
        total_batches = (len(unique_inchikeys) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(unique_inchikeys))
            batch_inchikeys = unique_inchikeys[start_idx:end_idx]

            if progress_callback:
                progress_callback(batch_idx + 1, total_batches)

            # Build VALUES clause with quoted InChIKeys
            inchikey_values = " ".join(f'"{ik}"' for ik in batch_inchikeys)
            query = QUERY_PUBCHEM_BY_INCHIKEY_TEMPLATE.format(
                inchikey_values=inchikey_values,
            )

            csv_bytes = execute_with_retry(query, endpoint, timeout=300)
            if not csv_bytes or len(csv_bytes) < 10:
                continue

            try:
                batch_df = pl.read_csv(
                    io.BytesIO(csv_bytes),
                    infer_schema_length=0,
                )
                if len(batch_df) > 0:
                    batch_df = batch_df.select(
                        [
                            pl.col("cid"),
                            pl.col("inchikey").alias("pubchem_inchikey"),
                            pl.col("mono_isotopic_weight").alias("pubchem_mass"),
                            pl.col("smiles").alias("pubchem_smiles"),
                            pl.col("connectivity_smiles").alias("pubchem_smiles_2d"),
                            pl.col("common_name"),
                            pl.col("iupac_name"),
                            pl.col("iupac_inchi").alias("pubchem_inchi"),
                            pl.col("molecular_formula").alias("pubchem_formula"),
                            pl.col("xlogp3").alias("pubchem_xlogp3"),
                            pl.col("defined_atom_stereo_count").alias(
                                "pubchem_defined_atom_stereo_count",
                            ),
                            pl.col("defined_bond_stereo_count").alias(
                                "pubchem_defined_bond_stereo_count",
                            ),
                            pl.col("undefined_atom_stereo_count").alias(
                                "pubchem_undefined_atom_stereo_count",
                            ),
                            pl.col("undefined_bond_stereo_count").alias(
                                "pubchem_undefined_bond_stereo_count",
                            ),
                        ],
                    )
                    all_results.append(batch_df)
            except Exception as e:
                print(
                    f"Warning: Could not parse PubChem InChIKey batch {batch_idx + 1}: {e}",
                    file=sys.stderr,
                )
                continue

        if not all_results:
            return pl.DataFrame(schema=schema)

        return pl.concat(all_results).unique(subset=["pubchem_inchikey"])

    # ========================================================================
    # DATA PROCESSING
    # ========================================================================

    # Subscript digit mapping for molecular formula normalization
    SUBSCRIPT_TO_NORMAL = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    def normalize_formula(formula: str | None) -> str | None:
        """Convert subscript digits to normal digits in molecular formulas.

        Parameters
        ----------
        formula : str | None
            Formula.

        Returns
        -------
        str | None
            ASCII digits.
        """
        if formula is None:
            return None
        return formula.translate(SUBSCRIPT_TO_NORMAL)

    def normalize_blank_strings(df: pl.DataFrame) -> pl.DataFrame:
        """Trim UTF-8 values and convert empty/whitespace-only cells and 'notClassified' to null.

        Parameters
        ----------
        df : pl.DataFrame
            Df.

        Returns
        -------
        pl.DataFrame
            UTF-8 text values and blanks set to null.
        """
        utf8_cols = [name for name, dtype in df.schema.items() if dtype == pl.Utf8]
        if not utf8_cols:
            return df

        return df.with_columns(
            [
                pl.col(col)
                .str.strip_chars()
                .replace(["", "notClassified"], [None, None])
                .alias(col)
                for col in utf8_cols
            ],
        )

    def extract_qids_from_lazyframe(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
        """Extract QIDs from Wikidata URLs in a column.

        Parameters
        ----------
        lf : pl.LazyFrame
            Lf.
        col : str
            Col.

        Returns
        -------
        pl.LazyFrame
            Wikidata entity prefixes removed from ``col``.
        """
        return lf.with_columns(
            pl.col(col)
            .str.replace(WIKIDATA_ENTITY_PREFIX, "", literal=True)
            .alias(col),
        )

    def collect_df(lf: pl.LazyFrame) -> pl.DataFrame:
        """Collect LazyFrame to DataFrame with proper type casting.

        Parameters
        ----------
        lf : pl.LazyFrame
            Lf.

        Returns
        -------
        pl.DataFrame
            DataFrame.
        """
        return cast(pl.DataFrame, lf.collect())

    def process_compound_taxon_reference(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Process compound-taxon-reference triplets.

        Parameters
        ----------
        lf : pl.LazyFrame
            Lf.

        Returns
        -------
        pl.LazyFrame
            QIDs for compound, taxon, and reference.
        """
        return (
            lf.pipe(extract_qids_from_lazyframe, "compound")
            .pipe(extract_qids_from_lazyframe, "taxon")
            .with_columns(
                pl.when(pl.col("reference").is_not_null())
                .then(
                    pl.col("reference").str.replace(
                        WIKIDATA_ENTITY_PREFIX,
                        "",
                        literal=True,
                    ),
                )
                .otherwise(pl.lit(None))
                .alias("reference"),
            )
        )

    # ========================================================================
    # OUTPUT BUILDING
    # ========================================================================

    def build_frozen_csv(
        compound_taxon_reference_df: pl.DataFrame,
        taxon_name_df: pl.DataFrame,
        reference_doi_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build frozen.csv output.

                                Columns:
                                - structure_inchikey
                                - organism_name
                                - reference_doi
                                - manual_validation (always NA)
                                - organism_wikidata
                                - structure_wikidata
                                - reference_wikidata

        Parameters
        ----------
        compound_taxon_reference_df : pl.DataFrame
            Compound taxon reference df.
        taxon_name_df : pl.DataFrame
            Taxon name df.
        reference_doi_df : pl.DataFrame
            Reference doi df.

        Returns
        -------
        pl.DataFrame
            Final ``frozen.csv`` table in the expected publication schema.
        """
        # Join with taxon names
        result = compound_taxon_reference_df.join(
            taxon_name_df.select(["taxon", "taxon_name"]),
            on="taxon",
            how="left",
        )

        # Join with reference DOIs
        result = result.join(
            reference_doi_df.select(["reference", "doi"]),
            on="reference",
            how="left",
        )

        # Build final output
        result = result.select(
            [
                pl.col("compound_inchikey").alias("structure_inchikey"),
                pl.col("taxon_name").alias("organism_name"),
                pl.col("doi").alias("reference_doi"),
                pl.lit(None).alias("manual_validation"),
                pl.concat_str([pl.lit(WIKIDATA_ENTITY_PREFIX), pl.col("taxon")]).alias(
                    "organism_wikidata",
                ),
                pl.concat_str(
                    [pl.lit(WIKIDATA_ENTITY_PREFIX), pl.col("compound")],
                ).alias("structure_wikidata"),
                pl.when(pl.col("reference").is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit(WIKIDATA_ENTITY_PREFIX), pl.col("reference")],
                    ),
                )
                .otherwise(pl.lit(None))
                .alias("reference_wikidata"),
            ],
        )

        return result

    def build_frozen_metadata_csv(
        compound_taxon_reference_df: pl.DataFrame,
        data: LOTUSExportData,
        npclassifier_df: pl.DataFrame,
        classyfire_df: pl.DataFrame,
        ott_df: pl.DataFrame,
        pubchem_df: pl.DataFrame,
        rdkit_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build frozen_metadata.csv output matching the old Zenodo format.

                                Data priority: PubChem > RDKit > Wikidata
                                - InChIKey, mass, SMILES from PubChem when available
                                - RDKit as fallback for properties PubChem doesn't provide (xlogp, stereocenters)
                                - Wikidata as final fallback

                                Columns:
                                - structure_wikidata, structure_inchikey, structure_inchi, structure_smiles
                                - structure_molecular_formula, structure_exact_mass, structure_xlogp
                                - structure_smiles_2D, structure_cid, structure_nameIupac, structure_nameTraditional
                                - structure_stereocenters_total, structure_stereocenters_unspecified
                                - structure_taxonomy_npclassifier_01pathway, _02superclass, _03class
                                - structure_taxonomy_classyfire_chemontid, _01kingdom, _02superclass, _03class, _04directparent
                                - organism_wikidata, organism_name
                                - organism_taxonomy_gbifid, organism_taxonomy_ncbiid, organism_taxonomy_ottid
                                - organism_taxonomy_01domain through organism_taxonomy_10varietas
                                - reference_wikidata, reference_doi, manual_validation

        Parameters
        ----------
        compound_taxon_reference_df : pl.DataFrame
            Compound taxon reference df.
        data : LOTUSExportData
            Data.
        npclassifier_df : pl.DataFrame
            Npclassifier df.
        classyfire_df : pl.DataFrame
            Classyfire df.
        ott_df : pl.DataFrame
            Ott df.
        pubchem_df : pl.DataFrame
            Pubchem df.
        rdkit_df : pl.DataFrame
            Rdkit df.

        Returns
        -------
        pl.DataFrame
            Final ``frozen_metadata.csv`` table in the legacy Zenodo format.
        """
        # Start with the triplets
        result = compound_taxon_reference_df.clone()

        # Collect all necessary metadata from Wikidata
        taxon_name_df = collect_df(
            extract_qids_from_lazyframe(data.taxon_name, "taxon"),
        )
        taxon_ncbi_df = collect_df(
            extract_qids_from_lazyframe(data.taxon_ncbi, "taxon"),
        )
        taxon_ott_df = collect_df(
            extract_qids_from_lazyframe(data.taxon_ott, "taxon"),
        )
        taxon_gbif_df = collect_df(
            extract_qids_from_lazyframe(data.taxon_gbif, "taxon"),
        )
        reference_doi_df = collect_df(
            extract_qids_from_lazyframe(data.reference_doi, "reference"),
        )
        smiles_iso_df = collect_df(
            extract_qids_from_lazyframe(data.compound_smiles_iso, "compound"),
        )
        smiles_can_df = collect_df(
            extract_qids_from_lazyframe(data.compound_smiles_can, "compound"),
        )
        cid_df = collect_df(extract_qids_from_lazyframe(data.compound_cid, "compound"))

        # Join taxon metadata
        result = (
            result.join(
                taxon_name_df.select(["taxon", "taxon_name"]),
                on="taxon",
                how="left",
            )
            .join(taxon_ncbi_df.select(["taxon", "taxon_ncbi"]), on="taxon", how="left")
            .join(taxon_ott_df.select(["taxon", "taxon_ott"]), on="taxon", how="left")
            .join(taxon_gbif_df.select(["taxon", "taxon_gbif"]), on="taxon", how="left")
        )

        # Join reference metadata
        result = result.join(
            reference_doi_df.select(["reference", "doi"]),
            on="reference",
            how="left",
        )

        # Join structure metadata from Wikidata
        result = (
            result.join(smiles_iso_df, on="compound", how="left")
            .join(smiles_can_df, on="compound", how="left")
            .join(cid_df, on="compound", how="left")
            .with_columns(
                pl.coalesce(
                    pl.col("compound_smiles_iso"),
                    pl.col("compound_smiles_can"),
                ).alias("_rdkit_join_smiles"),
            )
        )

        # Join NPClassifier data via SMILES (isomeric first, then canonical)
        if len(npclassifier_df) > 0:
            npc_renamed = npclassifier_df.filter(
                (pl.col("error").is_null() | (pl.col("error") == ""))
                & pl.col("pathway").is_not_null(),
            ).select(
                [
                    pl.col("smiles"),
                    pl.col("pathway").alias("npc_pathway"),
                    pl.col("superclass").alias("npc_superclass"),
                    pl.col("class").alias("npc_class"),
                ],
            )

            # Pass 1: join on isomeric SMILES
            result = result.join(
                npc_renamed,
                left_on="compound_smiles_iso",
                right_on="smiles",
                how="left",
            )

            # Pass 2: for rows that still have no NPC data, try canonical SMILES
            still_missing_npc = (
                pl.col("npc_pathway").is_null()
                & pl.col("compound_smiles_can").is_not_null()
            )
            result = (
                result.join(
                    npc_renamed.rename(
                        {
                            "smiles": "_npc2_smiles",
                            "npc_pathway": "_npc2_pathway",
                            "npc_superclass": "_npc2_superclass",
                            "npc_class": "_npc2_class",
                        },
                    ),
                    left_on="compound_smiles_can",
                    right_on="_npc2_smiles",
                    how="left",
                )
                .with_columns(
                    [
                        pl.coalesce("npc_pathway", "_npc2_pathway").alias(
                            "npc_pathway",
                        ),
                        pl.coalesce("npc_superclass", "_npc2_superclass").alias(
                            "npc_superclass",
                        ),
                        pl.coalesce("npc_class", "_npc2_class").alias("npc_class"),
                    ],
                )
                .drop(
                    ["_npc2_pathway", "_npc2_superclass", "_npc2_class"],
                    strict=False,
                )
            )

            # Pass 3: for rows that still have no NPC data, try the _rdkit_join_smiles
            # (coalesced iso/can used as the RDKit join key — catches SMILES normalisation differences)
            result = (
                result.join(
                    npc_renamed.rename(
                        {
                            "smiles": "_npc3_smiles",
                            "npc_pathway": "_npc3_pathway",
                            "npc_superclass": "_npc3_superclass",
                            "npc_class": "_npc3_class",
                        },
                    ),
                    left_on="_rdkit_join_smiles",
                    right_on="_npc3_smiles",
                    how="left",
                )
                .with_columns(
                    [
                        pl.coalesce("npc_pathway", "_npc3_pathway").alias(
                            "npc_pathway",
                        ),
                        pl.coalesce("npc_superclass", "_npc3_superclass").alias(
                            "npc_superclass",
                        ),
                        pl.coalesce("npc_class", "_npc3_class").alias("npc_class"),
                    ],
                )
                .drop(
                    ["_npc3_pathway", "_npc3_superclass", "_npc3_class"],
                    strict=False,
                )
            )
        else:
            result = result.with_columns(
                [
                    pl.lit(None).alias("npc_pathway"),
                    pl.lit(None).alias("npc_superclass"),
                    pl.lit(None).alias("npc_class"),
                ],
            )

        # Join ClassyFire data via InChIKey
        if len(classyfire_df) > 0:
            cf_renamed = classyfire_df.select(
                [
                    pl.col("inchikey"),
                    pl.col("chemontid").alias("cf_chemontid"),
                    pl.col("kingdom").alias("cf_kingdom"),
                    pl.col("superclass").alias("cf_superclass"),
                    pl.col("class").alias("cf_class"),
                    pl.col("direct_parent").alias("cf_directparent"),
                ],
            )
            result = result.join(
                cf_renamed,
                left_on="compound_inchikey",
                right_on="inchikey",
                how="left",
            )
        else:
            result = result.with_columns(
                [
                    pl.lit(None).alias("cf_chemontid"),
                    pl.lit(None).alias("cf_kingdom"),
                    pl.lit(None).alias("cf_superclass"),
                    pl.lit(None).alias("cf_class"),
                    pl.lit(None).alias("cf_directparent"),
                ],
            )

        # Join OTT taxonomy via organism_name with smart fallback logic
        if len(ott_df) > 0:
            result = result.with_columns(
                [
                    pl.col("taxon_name")
                    .cast(pl.Utf8)
                    .str.strip_chars()
                    .alias("_taxon_name_key"),
                    pl.col("taxon_ott")
                    .cast(pl.Utf8)
                    .str.strip_chars()
                    .alias("_taxon_ott_key"),
                ],
            )

            ott_renamed = ott_df.select(
                [
                    pl.col("organism_name")
                    .cast(pl.Utf8)
                    .str.strip_chars()
                    .alias("_ott_name_key"),
                    pl.col("ott_id")
                    .cast(pl.Utf8)
                    .str.strip_chars()
                    .alias("_ott_id_key"),
                    pl.col("ott_id"),
                    pl.col("domain").alias("ott_domain"),
                    pl.col("kingdom").alias("ott_kingdom"),
                    pl.col("phylum").alias("ott_phylum"),
                    pl.col("class").alias("ott_class"),
                    pl.col("order").alias("ott_order"),
                    pl.col("family").alias("ott_family"),
                    pl.col("tribe").alias("ott_tribe"),
                    pl.col("genus").alias("ott_genus"),
                    pl.col("species").alias("ott_species"),
                    pl.col("varietas").alias("ott_varietas"),
                ],
            )

            # Strategy: join in two phases
            # Phase 1: Join rows where both taxon_ott exists in original DF - join by organism_name AND ott_id
            # Phase 2: For rows without match, join by organism_name only

            # Identify rows that have ott_id from Wikidata
            has_wikidata_ott = result.filter(pl.col("_taxon_ott_key").is_not_null())
            no_wikidata_ott = result.filter(pl.col("_taxon_ott_key").is_null())

            # Phase 1: Join on both organism_name and ott_id
            if len(has_wikidata_ott) > 0:
                result_with_ott = has_wikidata_ott.join(
                    ott_renamed,
                    left_on=["_taxon_name_key", "_taxon_ott_key"],
                    right_on=["_ott_name_key", "_ott_id_key"],
                    how="left",
                )
            else:
                result_with_ott = has_wikidata_ott

            # Phase 2: Join on organism_name only for rows without Wikidata ott_id
            if len(no_wikidata_ott) > 0:
                result_no_ott = no_wikidata_ott.join(
                    ott_renamed,
                    left_on="_taxon_name_key",
                    right_on="_ott_name_key",
                    how="left",
                )
            else:
                result_no_ott = no_wikidata_ott

            # Ensure identical schemas before concat; helper right-side key
            # columns can differ by branch depending on join keys.
            result_with_ott = result_with_ott.drop(["_ott_id_key"], strict=False)
            result_no_ott = result_no_ott.drop(["_ott_id_key"], strict=False)

            # Combine both results
            result = pl.concat([result_with_ott, result_no_ott]).drop(
                ["_taxon_name_key", "_taxon_ott_key"],
                strict=False,
            )
        else:
            result = result.with_columns(
                [
                    pl.lit(None).alias("ott_id"),
                    pl.lit(None).alias("ott_domain"),
                    pl.lit(None).alias("ott_kingdom"),
                    pl.lit(None).alias("ott_phylum"),
                    pl.lit(None).alias("ott_class"),
                    pl.lit(None).alias("ott_order"),
                    pl.lit(None).alias("ott_family"),
                    pl.lit(None).alias("ott_tribe"),
                    pl.lit(None).alias("ott_genus"),
                    pl.lit(None).alias("ott_species"),
                    pl.lit(None).alias("ott_varietas"),
                ],
            )

        # Join PubChem compound data via CID first, then InChIKey fallback
        # This maximizes coverage since not all compounds have CID in Wikidata
        if len(pubchem_df) > 0:
            # First join by CID
            result = result.join(
                pubchem_df,
                left_on="compound_cid",
                right_on="cid",
                how="left",
            )

            # For rows without PubChem data, try joining by InChIKey
            # Create a lookup by InChIKey for fallback
            pubchem_by_inchikey = pubchem_df.select(
                [
                    pl.col("pubchem_inchikey").alias("_pk_inchikey"),
                    pl.col("pubchem_mass").alias("_pk_mass"),
                    pl.col("pubchem_smiles").alias("_pk_smiles"),
                    pl.col("pubchem_smiles_2d").alias("_pk_smiles_2d"),
                    pl.col("common_name").alias("_pk_common_name"),
                    pl.col("iupac_name").alias("_pk_iupac_name"),
                    pl.col("pubchem_inchi").alias("_pk_inchi"),
                    pl.col("pubchem_formula").alias("_pk_formula"),
                    pl.col("pubchem_xlogp3").alias("_pk_xlogp3"),
                    pl.col("pubchem_defined_atom_stereo_count").alias(
                        "_pk_defined_atom_stereo_count",
                    ),
                    pl.col("pubchem_defined_bond_stereo_count").alias(
                        "_pk_defined_bond_stereo_count",
                    ),
                    pl.col("pubchem_undefined_atom_stereo_count").alias(
                        "_pk_undefined_atom_stereo_count",
                    ),
                    pl.col("pubchem_undefined_bond_stereo_count").alias(
                        "_pk_undefined_bond_stereo_count",
                    ),
                ],
            ).unique(subset=["_pk_inchikey"])

            result = result.join(
                pubchem_by_inchikey,
                left_on="compound_inchikey",
                right_on="_pk_inchikey",
                how="left",
            )

            # Coalesce: prefer CID-matched data, fall back to InChIKey-matched
            result = result.with_columns(
                [
                    pl.coalesce(
                        pl.col("pubchem_inchikey"),
                        pl.col("compound_inchikey"),
                    ).alias("pubchem_inchikey"),
                    pl.coalesce(pl.col("pubchem_mass"), pl.col("_pk_mass")).alias(
                        "pubchem_mass",
                    ),
                    pl.coalesce(pl.col("pubchem_smiles"), pl.col("_pk_smiles")).alias(
                        "pubchem_smiles",
                    ),
                    pl.coalesce(
                        pl.col("pubchem_smiles_2d"),
                        pl.col("_pk_smiles_2d"),
                    ).alias("pubchem_smiles_2d"),
                    pl.coalesce(pl.col("common_name"), pl.col("_pk_common_name")).alias(
                        "common_name",
                    ),
                    pl.coalesce(pl.col("iupac_name"), pl.col("_pk_iupac_name")).alias(
                        "iupac_name",
                    ),
                    pl.coalesce(pl.col("pubchem_inchi"), pl.col("_pk_inchi")).alias(
                        "pubchem_inchi",
                    ),
                    pl.coalesce(pl.col("pubchem_formula"), pl.col("_pk_formula")).alias(
                        "pubchem_formula",
                    ),
                    pl.coalesce(pl.col("pubchem_xlogp3"), pl.col("_pk_xlogp3")).alias(
                        "pubchem_xlogp3",
                    ),
                    pl.coalesce(
                        pl.col("pubchem_defined_atom_stereo_count"),
                        pl.col("_pk_defined_atom_stereo_count"),
                    ).alias("pubchem_defined_atom_stereo_count"),
                    pl.coalesce(
                        pl.col("pubchem_defined_bond_stereo_count"),
                        pl.col("_pk_defined_bond_stereo_count"),
                    ).alias("pubchem_defined_bond_stereo_count"),
                    pl.coalesce(
                        pl.col("pubchem_undefined_atom_stereo_count"),
                        pl.col("_pk_undefined_atom_stereo_count"),
                    ).alias("pubchem_undefined_atom_stereo_count"),
                    pl.coalesce(
                        pl.col("pubchem_undefined_bond_stereo_count"),
                        pl.col("_pk_undefined_bond_stereo_count"),
                    ).alias("pubchem_undefined_bond_stereo_count"),
                ],
            ).drop(
                [
                    "_pk_mass",
                    "_pk_smiles",
                    "_pk_smiles_2d",
                    "_pk_common_name",
                    "_pk_iupac_name",
                    "_pk_inchi",
                    "_pk_formula",
                    "_pk_xlogp3",
                    "_pk_defined_atom_stereo_count",
                    "_pk_defined_bond_stereo_count",
                    "_pk_undefined_atom_stereo_count",
                    "_pk_undefined_bond_stereo_count",
                ],
            )
        else:
            result = result.with_columns(
                [
                    pl.lit(None).alias("pubchem_inchikey"),
                    pl.lit(None).alias("pubchem_mass"),
                    pl.lit(None).alias("pubchem_smiles"),
                    pl.lit(None).alias("pubchem_smiles_2d"),
                    pl.lit(None).alias("common_name"),
                    pl.lit(None).alias("iupac_name"),
                    pl.lit(None).alias("pubchem_inchi"),
                    pl.lit(None).alias("pubchem_formula"),
                    pl.lit(None).alias("pubchem_xlogp3"),
                    pl.lit(None).alias("pubchem_defined_atom_stereo_count"),
                    pl.lit(None).alias("pubchem_defined_bond_stereo_count"),
                    pl.lit(None).alias("pubchem_undefined_atom_stereo_count"),
                    pl.lit(None).alias("pubchem_undefined_bond_stereo_count"),
                ],
            )

        # Join RDKit-derived data via SMILES (fallback for xlogp, stereocenters, and missing data)
        if len(rdkit_df) > 0:
            rdkit_renamed = rdkit_df.select(
                [
                    pl.col("smiles"),
                    pl.col("smiles_2D").alias("rdkit_smiles_2d"),
                    pl.col("molecular_formula").alias("rdkit_formula"),
                    pl.col("exact_mass").alias("rdkit_mass"),
                    pl.col("xlogp").alias("rdkit_xlogp"),
                    pl.col("stereocenters_total").alias("rdkit_stereo_total"),
                    pl.col("stereocenters_unspecified").alias("rdkit_stereo_unspec"),
                    pl.col("inchi").alias("rdkit_inchi"),
                    pl.col("inchikey").alias("rdkit_inchikey"),
                ],
            )
            result = result.join(
                rdkit_renamed,
                left_on="_rdkit_join_smiles",
                right_on="smiles",
                how="left",
            )
        else:
            result = result.with_columns(
                [
                    pl.lit(None).alias("rdkit_smiles_2d"),
                    pl.lit(None).alias("rdkit_formula"),
                    pl.lit(None).alias("rdkit_mass"),
                    pl.lit(None).alias("rdkit_xlogp"),
                    pl.lit(None).alias("rdkit_stereo_total"),
                    pl.lit(None).alias("rdkit_stereo_unspec"),
                    pl.lit(None).alias("rdkit_inchi"),
                    pl.lit(None).alias("rdkit_inchikey"),
                ],
            )

        # Add manual_validation column
        result = result.with_columns(pl.lit("NA").alias("manual_validation"))

        # Harmonize stereocenter counts: PubChem (atom+bond) -> fallback RDKit
        result = result.with_columns(
            [
                pl.when(
                    pl.any_horizontal(
                        [
                            pl.col("pubchem_defined_atom_stereo_count").is_not_null(),
                            pl.col("pubchem_defined_bond_stereo_count").is_not_null(),
                            pl.col("pubchem_undefined_atom_stereo_count").is_not_null(),
                            pl.col("pubchem_undefined_bond_stereo_count").is_not_null(),
                        ],
                    ),
                )
                .then(
                    pl.sum_horizontal(
                        [
                            pl.col("pubchem_defined_atom_stereo_count").cast(
                                pl.Int64,
                                strict=False,
                            ),
                            pl.col("pubchem_defined_bond_stereo_count").cast(
                                pl.Int64,
                                strict=False,
                            ),
                            pl.col("pubchem_undefined_atom_stereo_count").cast(
                                pl.Int64,
                                strict=False,
                            ),
                            pl.col("pubchem_undefined_bond_stereo_count").cast(
                                pl.Int64,
                                strict=False,
                            ),
                        ],
                    ).cast(pl.Utf8),
                )
                .otherwise(pl.lit(None))
                .alias("pubchem_stereo_total"),
                pl.when(
                    pl.any_horizontal(
                        [
                            pl.col("pubchem_undefined_atom_stereo_count").is_not_null(),
                            pl.col("pubchem_undefined_bond_stereo_count").is_not_null(),
                        ],
                    ),
                )
                .then(
                    pl.sum_horizontal(
                        [
                            pl.col("pubchem_undefined_atom_stereo_count").cast(
                                pl.Int64,
                                strict=False,
                            ),
                            pl.col("pubchem_undefined_bond_stereo_count").cast(
                                pl.Int64,
                                strict=False,
                            ),
                        ],
                    ).cast(pl.Utf8),
                )
                .otherwise(pl.lit(None))
                .alias("pubchem_stereo_unspecified"),
            ],
        )

        # Select final columns in expected order (matching old Zenodo format)
        # Priority: PubChem > RDKit > Wikidata
        result = result.select(
            [
                # Structure identifiers
                pl.concat_str(
                    [pl.lit(WIKIDATA_ENTITY_PREFIX), pl.col("compound")],
                ).alias("structure_wikidata"),
                # InChIKey: PubChem > RDKit > Wikidata
                pl.coalesce(
                    pl.col("pubchem_inchikey"),
                    pl.col("rdkit_inchikey"),
                    pl.col("compound_inchikey"),
                ).alias("structure_inchikey"),
                # InChI: PubChem > RDKit
                pl.coalesce(pl.col("pubchem_inchi"), pl.col("rdkit_inchi")).alias(
                    "structure_inchi",
                ),
                # SMILES: PubChem > Wikidata isomeric
                pl.coalesce(
                    pl.col("pubchem_smiles"),
                    pl.col("compound_smiles_iso"),
                    pl.col("compound_smiles_can"),
                    pl.col("_rdkit_join_smiles"),
                ).alias("structure_smiles"),
                # Molecular formula: RDKit only, normalized to remove subscript digits
                pl.coalesce(pl.col("pubchem_formula"), pl.col("rdkit_formula"))
                .map_elements(normalize_formula, return_dtype=pl.Utf8)
                .alias("structure_molecular_formula"),
                # Mass: PubChem > RDKit
                pl.coalesce(
                    pl.col("pubchem_mass"),
                    pl.col("rdkit_mass"),
                ).alias("structure_exact_mass"),
                # XLogP: PubChem xlogp3 > RDKit
                pl.coalesce(pl.col("pubchem_xlogp3"), pl.col("rdkit_xlogp")).alias(
                    "structure_xlogp",
                ),
                # 2D SMILES: PubChem connectivity > RDKit > Wikidata canonical
                pl.coalesce(
                    pl.col("pubchem_smiles_2d"),
                    pl.col("rdkit_smiles_2d"),
                    pl.col("compound_smiles_can"),
                ).alias("structure_smiles_2D"),
                pl.col("compound_cid").alias("structure_cid"),
                pl.col("iupac_name").alias("structure_nameIupac"),
                pl.col("common_name").alias("structure_nameTraditional"),
                # Stereocenters: PubChem (defined+undefined atom/bond) > RDKit
                pl.coalesce(
                    pl.col("pubchem_stereo_total"),
                    pl.col("rdkit_stereo_total"),
                ).alias("structure_stereocenters_total"),
                pl.coalesce(
                    pl.col("pubchem_stereo_unspecified"),
                    pl.col("rdkit_stereo_unspec"),
                ).alias(
                    "structure_stereocenters_unspecified",
                ),
                # NPClassifier taxonomy
                pl.col("npc_pathway").alias(
                    "structure_taxonomy_npclassifier_01pathway",
                ),
                pl.col("npc_superclass").alias(
                    "structure_taxonomy_npclassifier_02superclass",
                ),
                pl.col("npc_class").alias("structure_taxonomy_npclassifier_03class"),
                # ClassyFire taxonomy
                pl.col("cf_chemontid").alias("structure_taxonomy_classyfire_chemontid"),
                pl.col("cf_kingdom").alias("structure_taxonomy_classyfire_01kingdom"),
                pl.col("cf_superclass").alias(
                    "structure_taxonomy_classyfire_02superclass",
                ),
                pl.col("cf_class").alias("structure_taxonomy_classyfire_03class"),
                pl.col("cf_directparent").alias(
                    "structure_taxonomy_classyfire_04directparent",
                ),
                # Organism identifiers
                pl.concat_str([pl.lit(WIKIDATA_ENTITY_PREFIX), pl.col("taxon")]).alias(
                    "organism_wikidata",
                ),
                pl.col("taxon_name").alias("organism_name"),
                pl.col("taxon_gbif").alias("organism_taxonomy_gbifid"),
                pl.col("taxon_ncbi").alias("organism_taxonomy_ncbiid"),
                pl.coalesce(pl.col("taxon_ott"), pl.col("ott_id")).alias(
                    "organism_taxonomy_ottid",
                ),
                # OTT taxonomy hierarchy
                pl.col("ott_domain").alias("organism_taxonomy_01domain"),
                pl.col("ott_kingdom").alias("organism_taxonomy_02kingdom"),
                pl.col("ott_phylum").alias("organism_taxonomy_03phylum"),
                pl.col("ott_class").alias("organism_taxonomy_04class"),
                pl.col("ott_order").alias("organism_taxonomy_05order"),
                pl.col("ott_family").alias("organism_taxonomy_06family"),
                pl.col("ott_tribe").alias("organism_taxonomy_07tribe"),
                pl.col("ott_genus").alias("organism_taxonomy_08genus"),
                pl.col("ott_species").alias("organism_taxonomy_09species"),
                pl.col("ott_varietas").alias("organism_taxonomy_10varietas"),
                # Reference identifiers
                pl.when(pl.col("reference").is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit(WIKIDATA_ENTITY_PREFIX), pl.col("reference")],
                    ),
                )
                .otherwise(pl.lit(None))
                .alias("reference_wikidata"),
                pl.col("doi").alias("reference_doi"),
                # Validation
                "manual_validation",
            ],
        )

        return normalize_blank_strings(result)


@app.cell
def md_title():
    mo.md("""
    # LOTUS Data Exporter

    This app generates CSV files matching the [Zenodo LOTUS data exports](https://zenodo.org/communities/the-lotus-initiative):
    - **frozen.csv**: Core triplets (structure_inchikey, organism_name, reference_doi)
    - **frozen_metadata.csv**: Full metadata for all triplets including:
      - Structure: InChIKey, InChI, SMILES, formula, mass, xlogp, stereocenters, PubChem CID
      - Structure taxonomy: NPClassifier pathway/superclass/class, ClassyFire hierarchy
      - Organism: name, GBIF ID, NCBI ID, OTT taxonomy hierarchy
      - Reference: DOI, Wikidata ID

    *Data is fetched from Wikidata and enriched with external caches.*
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

                The LOTUS Data Exporter processes large amounts of data from Wikidata,
                which exceeds WebAssembly memory limits.

                **Please use the CLI instead:**
                ```bash
                uv run https://adafede.github.io/marimo/apps/lotus_exporter.py export -o ./output -v
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
    import time

    mo.stop(not run_button.value, mo.md("Click **Fetch Data from Wikidata** to start"))

    start_time = time.time()

    with mo.status.spinner("Fetching data from Wikidata...") as _spinner:

        def progress_callback(msg):
            _spinner.update(msg)

        data = fetch_all_data(CONFIG["qlever_endpoint"], progress_callback)

    with mo.status.spinner("Fetching NPClassifier cache..."):
        npclassifier_df = fetch_npclassifier_cache()

    with mo.status.spinner("Fetching ClassyFire cache..."):
        classyfire_df = fetch_classyfire_cache()

    with mo.status.spinner("Fetching OTT taxonomy cache..."):
        ott_df = fetch_ott_taxonomy_cache()

    # Get unique SMILES and CIDs for processing
    with mo.status.spinner("Processing structure data..."):
        smiles_iso_df = collect_df(
            extract_qids_from_lazyframe(data.compound_smiles_iso, "compound"),
        )
        smiles_can_df = collect_df(
            extract_qids_from_lazyframe(data.compound_smiles_can, "compound"),
        )
        cid_df = collect_df(
            extract_qids_from_lazyframe(data.compound_cid, "compound"),
        )
        # Get all unique InChIKeys from compound_taxon_reference (with QIDs extracted)
        _triplets_df = collect_df(
            extract_qids_from_lazyframe(data.compound_taxon_reference, "compound"),
        )
        all_inchikeys = (
            _triplets_df.select("compound_inchikey").unique().to_series().to_list()
        )
        unique_cids = (
            cid_df.select("compound_cid").unique().drop_nulls().to_series().to_list()
        )

    # Fetch PubChem data by CID
    total_batches = (len(unique_cids) + PUBCHEM_BATCH_SIZE - 1) // PUBCHEM_BATCH_SIZE
    with mo.status.spinner(
        f"Fetching PubChem data for {len(unique_cids):,} CIDs in {total_batches} batches...",
    ) as _pubchem_spinner:

        def pubchem_progress(current, total):
            _pubchem_spinner.update(
                f"Fetching PubChem data: batch {current}/{total} "
                f"({current * PUBCHEM_BATCH_SIZE:,}/{len(unique_cids):,} CIDs)...",
            )

        pubchem_df = fetch_pubchem_compound_data(unique_cids, pubchem_progress)

    # Find InChIKeys without CID matches and fetch from PubChem by InChIKey
    matched_inchikeys = (
        set(pubchem_df["pubchem_inchikey"].to_list()) if len(pubchem_df) > 0 else set()
    )
    missing_inchikeys = [
        ik for ik in all_inchikeys if ik and ik not in matched_inchikeys
    ]

    if missing_inchikeys:
        with mo.status.spinner(
            f"Fetching PubChem data for {len(missing_inchikeys):,} InChIKeys without CID...",
        ) as _ik_spinner:

            def ik_progress(current, total):
                _ik_spinner.update(
                    f"Fetching PubChem by InChIKey: batch {current}/{total}...",
                )

            pubchem_ik_df = fetch_pubchem_by_inchikey(missing_inchikeys, ik_progress)

            if len(pubchem_ik_df) > 0:
                # Merge with existing pubchem_df
                pubchem_df = pl.concat([pubchem_df, pubchem_ik_df]).unique(
                    subset=["pubchem_inchikey"],
                )

    # Only compute RDKit for SMILES without PubChem data (fallback)
    matched_inchikeys_final = (
        set(pubchem_df["pubchem_inchikey"].to_list()) if len(pubchem_df) > 0 else set()
    )
    # Get SMILES for compounds that don't have PubChem data
    # First, find compounds (by InChIKey) without PubChem matches
    compounds_needing_rdkit = (
        _triplets_df.filter(
            ~pl.col("compound_inchikey").is_in(list(matched_inchikeys_final)),
        )
        .select("compound")
        .unique()
    )

    # Then get SMILES (isomeric preferred, canonical fallback)
    smiles_needing_rdkit = (
        pl.concat(
            [
                smiles_iso_df.join(
                    compounds_needing_rdkit,
                    on="compound",
                    how="inner",
                ).select(pl.col("compound_smiles_iso").alias("smiles")),
                smiles_can_df.join(
                    compounds_needing_rdkit,
                    on="compound",
                    how="inner",
                ).select(pl.col("compound_smiles_can").alias("smiles")),
            ],
            how="vertical",
        )
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )

    if smiles_needing_rdkit:
        with mo.status.spinner(
            f"Computing RDKit properties for {len(smiles_needing_rdkit):,} structures without PubChem data...",
        ):
            rdkit_df = compute_rdkit_properties(smiles_needing_rdkit)
    else:
        rdkit_df = pl.DataFrame()

    elapsed = round(time.time() - start_time, 2)
    mo.md(f"Data fetched in **{elapsed}s**")
    return classyfire_df, data, npclassifier_df, ott_df, pubchem_df, rdkit_df


@app.cell
def display_stats(
    classyfire_df,
    data,
    npclassifier_df,
    ott_df,
    pubchem_df,
    rdkit_df,
):
    mo.stop(data is None)

    # Collect stats
    _triplets_df_stats = data.compound_taxon_reference.collect()
    n_triplets = len(_triplets_df_stats)
    n_compounds = _triplets_df_stats.select(pl.col("compound").n_unique()).item()
    n_taxa = _triplets_df_stats.select(pl.col("taxon").n_unique()).item()
    n_refs = (
        _triplets_df_stats.filter(pl.col("reference").is_not_null())
        .select(pl.col("reference").n_unique())
        .item()
    )

    mo.vstack(
        [
            mo.md("## Data Statistics"),
            mo.hstack(
                [
                    mo.stat(value=f"{n_triplets:,}", label="Total Triplets"),
                    mo.stat(value=f"{n_compounds:,}", label="Unique Compounds"),
                    mo.stat(value=f"{n_taxa:,}", label="Unique Taxa"),
                    mo.stat(value=f"{n_refs:,}", label="Unique References"),
                ],
                gap=0,
                wrap=True,
            ),
            mo.md("### External Caches & Computed Data"),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{len(npclassifier_df):,}",
                        label="NPClassifier",
                    ),
                    mo.stat(
                        value=f"{len(classyfire_df):,}",
                        label="ClassyFire",
                    ),
                    mo.stat(
                        value=f"{len(ott_df):,}",
                        label="OTT Taxonomy",
                    ),
                    mo.stat(
                        value=f"{len(pubchem_df):,}",
                        label="PubChem Data",
                    ),
                    mo.stat(
                        value=f"{len(rdkit_df):,}",
                        label="RDKit (fallback)",
                    ),
                ],
                gap=0,
                wrap=True,
            ),
        ],
    )
    return


@app.cell
def footer():
    mo.md("""
    ---
    **Data:**
    <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> &
    <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a> |
    **Zenodo:**
    <a href="https://zenodo.org/communities/the-lotus-initiative" style="color:#006699;">LOTUS Community</a> |
    **Code:**
    <a href="https://adafede.github.io/marimo/apps/lotus_exporter.py" style="color:#339966;">lotus_exporter.py</a> |
    **License:**
    <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#484848;">CC0 1.0</a> for data &
    <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#484848;">AGPL-3.0</a> for code
    """)
    return


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """Entry point for CLI mode."""
    import argparse
    from datetime import datetime
    from pathlib import Path

    # Check if running as CLI with 'export' command
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        parser = argparse.ArgumentParser(
            description="LOTUS Data Exporter - Export LOTUS data from Wikidata",
            prog="lotus_exporter.py export",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default="./output",
            help="Output directory (default: ./output)",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Verbose output",
        )
        args = parser.parse_args(sys.argv[2:])

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if args.verbose:
                print("=" * 60, file=sys.stderr)
                print("LOTUS Data Exporter - CLI Export", file=sys.stderr)
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
                print("\nFetching external caches...", file=sys.stderr)
                print("  → NPClassifier cache...", file=sys.stderr)
            npclassifier_df = fetch_npclassifier_cache()

            if args.verbose:
                print("  → ClassyFire cache...", file=sys.stderr)
            classyfire_df = fetch_classyfire_cache()

            if args.verbose:
                print("  → OTT taxonomy cache...", file=sys.stderr)
            ott_df = fetch_ott_taxonomy_cache()

            if args.verbose:
                print("\nFetching previous version from Zenodo...", file=sys.stderr)

            # Fetch previous frozen.csv from Zenodo for manual_validation inheritance
            old_frozen_df = fetch_latest_zenodo_frozen()
            if args.verbose:
                if len(old_frozen_df) > 0:
                    n_validated = len(
                        old_frozen_df.filter(pl.col("manual_validation") != "NA"),
                    )
                    print(
                        f"  Loaded {len(old_frozen_df):,} rows from Zenodo ({n_validated} validated)",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "  No previous frozen.csv available from Zenodo",
                        file=sys.stderr,
                    )

            if args.verbose:
                print("\nProcessing data...", file=sys.stderr)

            # Process compound-taxon-reference triplets
            compound_taxon_reference_df = collect_df(
                process_compound_taxon_reference(data.compound_taxon_reference),
            )

            if args.verbose:
                print(
                    f"  Triplets: {len(compound_taxon_reference_df):,}",
                    file=sys.stderr,
                )

            # Get unique SMILES and CIDs for PubChem and RDKit processing
            smiles_iso_df = collect_df(
                extract_qids_from_lazyframe(data.compound_smiles_iso, "compound"),
            )
            smiles_can_df = collect_df(
                extract_qids_from_lazyframe(data.compound_smiles_can, "compound"),
            )
            cid_df = collect_df(
                extract_qids_from_lazyframe(data.compound_cid, "compound"),
            )
            # Get all unique InChIKeys
            all_inchikeys = (
                compound_taxon_reference_df.select("compound_inchikey")
                .unique()
                .to_series()
                .to_list()
            )
            unique_cids = (
                cid_df.select("compound_cid")
                .unique()
                .drop_nulls()
                .to_series()
                .to_list()
            )

            # Fetch PubChem compound data (primary source) in batches
            total_batches = (
                len(unique_cids) + PUBCHEM_BATCH_SIZE - 1
            ) // PUBCHEM_BATCH_SIZE
            if args.verbose:
                print(
                    f"  → Fetching PubChem data for {len(unique_cids):,} CIDs in {total_batches} batches...",
                    file=sys.stderr,
                )

            def cli_pubchem_progress(current, total):
                if args.verbose:
                    print(
                        f"     Batch {current}/{total} ({current * PUBCHEM_BATCH_SIZE:,} CIDs processed)",
                        file=sys.stderr,
                    )

            pubchem_df = fetch_pubchem_compound_data(unique_cids, cli_pubchem_progress)
            if args.verbose and len(pubchem_df) > 0:
                print(
                    f"     Fetched data for {len(pubchem_df):,} compounds by CID",
                    file=sys.stderr,
                )

            # Fetch PubChem data by InChIKey for compounds without CIDs
            matched_inchikeys = (
                set(pubchem_df["pubchem_inchikey"].to_list())
                if len(pubchem_df) > 0
                else set()
            )
            missing_inchikeys = [
                ik for ik in all_inchikeys if ik and ik not in matched_inchikeys
            ]

            if missing_inchikeys:
                total_ik_batches = (
                    len(missing_inchikeys) + PUBCHEM_BATCH_SIZE - 1
                ) // PUBCHEM_BATCH_SIZE
                if args.verbose:
                    print(
                        f"  → Fetching PubChem data for {len(missing_inchikeys):,} InChIKeys without CID in {total_ik_batches} batches...",
                        file=sys.stderr,
                    )

                def cli_ik_progress(current, total):
                    if args.verbose:
                        print(
                            f"     InChIKey batch {current}/{total}",
                            file=sys.stderr,
                        )

                pubchem_ik_df = fetch_pubchem_by_inchikey(
                    missing_inchikeys,
                    cli_ik_progress,
                )
                if len(pubchem_ik_df) > 0:
                    if args.verbose:
                        print(
                            f"     Fetched data for {len(pubchem_ik_df):,} additional compounds by InChIKey",
                            file=sys.stderr,
                        )
                    pubchem_df = pl.concat([pubchem_df, pubchem_ik_df]).unique(
                        subset=["pubchem_inchikey"],
                    )

            # Compute RDKit properties only for compounds without PubChem data
            matched_inchikeys_final = (
                set(pubchem_df["pubchem_inchikey"].to_list())
                if len(pubchem_df) > 0
                else set()
            )
            compounds_needing_rdkit = (
                compound_taxon_reference_df.filter(
                    ~pl.col("compound_inchikey").is_in(list(matched_inchikeys_final)),
                )
                .select("compound")
                .unique()
            )

            smiles_needing_rdkit = (
                pl.concat(
                    [
                        smiles_iso_df.join(
                            compounds_needing_rdkit,
                            on="compound",
                            how="inner",
                        ).select(pl.col("compound_smiles_iso").alias("smiles")),
                        smiles_can_df.join(
                            compounds_needing_rdkit,
                            on="compound",
                            how="inner",
                        ).select(pl.col("compound_smiles_can").alias("smiles")),
                    ],
                    how="vertical",
                )
                .unique()
                .drop_nulls()
                .to_series()
                .to_list()
            )

            if smiles_needing_rdkit:
                if args.verbose:
                    print(
                        f"  → Computing RDKit properties for {len(smiles_needing_rdkit):,} SMILES without PubChem data...",
                        file=sys.stderr,
                    )
                rdkit_df = compute_rdkit_properties(smiles_needing_rdkit)
            else:
                rdkit_df = pl.DataFrame()
            if args.verbose and len(rdkit_df) > 0:
                print(
                    f"     Computed properties for {len(rdkit_df):,} structures",
                    file=sys.stderr,
                )

            # Build outputs
            if args.verbose:
                print("\nBuilding output files...", file=sys.stderr)

            date_str = datetime.now().strftime("%y%m%d")

            # Build frozen.csv
            if args.verbose:
                print("  → Building frozen.csv...", file=sys.stderr)
            taxon_name_df = collect_df(
                extract_qids_from_lazyframe(data.taxon_name, "taxon"),
            )
            reference_doi_df = collect_df(
                extract_qids_from_lazyframe(data.reference_doi, "reference"),
            )
            frozen_df = build_frozen_csv(
                compound_taxon_reference_df,
                taxon_name_df,
                reference_doi_df,
            )

            # Inherit manual_validation from previous version
            if len(old_frozen_df) > 0:
                if args.verbose:
                    print(
                        "  → Inheriting manual_validation from previous version...",
                        file=sys.stderr,
                    )
                frozen_df = inherit_manual_validation(frozen_df, old_frozen_df)
                n_inherited = len(frozen_df.filter(pl.col("manual_validation") != "NA"))
                if args.verbose:
                    print(
                        f"     Inherited {n_inherited} validated entries",
                        file=sys.stderr,
                    )

            # Build frozen_metadata.csv (comprehensive)
            if args.verbose:
                print("  → Building frozen_metadata.csv...", file=sys.stderr)
            frozen_metadata_df = build_frozen_metadata_csv(
                compound_taxon_reference_df,
                data,
                npclassifier_df,
                classyfire_df,
                ott_df,
                pubchem_df,
                rdkit_df,
            )
            frozen_metadata_df = drop_missing_structure_rows(
                frozen_metadata_df,
                verbose=args.verbose,
            )
            if len(old_frozen_df) > 0:
                frozen_metadata_df = inherit_manual_validation(
                    frozen_metadata_df,
                    old_frozen_df,
                )

            # Final cleanup: trim and nullify blank/whitespace UTF-8 values
            frozen_df = normalize_blank_strings(frozen_df)
            frozen_metadata_df = normalize_blank_strings(frozen_metadata_df)

            if args.verbose:
                print("\nWriting output files...", file=sys.stderr)

            # Write frozen.csv.gz
            frozen_path = output_dir / f"{date_str}_frozen.csv.gz"
            frozen_df.write_csv(frozen_path, compression="gzip")
            if args.verbose:
                print(f"  ✓ {frozen_path}", file=sys.stderr)
            else:
                print(frozen_path)

            # Write frozen_metadata.csv.gz
            frozen_metadata_path = output_dir / f"{date_str}_frozen_metadata.csv.gz"
            frozen_metadata_df.write_csv(frozen_metadata_path, compression="gzip")
            if args.verbose:
                print(f"  ✓ {frozen_metadata_path}", file=sys.stderr)
            else:
                print(frozen_metadata_path)

            # Compute and write changes report
            if args.verbose:
                print("\nComputing changes from previous version...", file=sys.stderr)

            changes = []

            # Frozen changes
            frozen_changes = compute_changes(
                frozen_df,
                old_frozen_df,
                ["structure_inchikey", "organism_wikidata", "reference_wikidata"],
                "Frozen (triplets)",
            )
            changes.append(frozen_changes)
            if args.verbose:
                print(
                    f"  Frozen: +{frozen_changes['added']:,} / -{frozen_changes['removed']:,}",
                    file=sys.stderr,
                )

            # Write changes report
            changes_report_path = output_dir / f"{date_str}_changes_report.txt"
            write_changes_report(changes, str(changes_report_path))
            if args.verbose:
                print(f"\n  ✓ {changes_report_path}", file=sys.stderr)
            else:
                print(changes_report_path)

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
