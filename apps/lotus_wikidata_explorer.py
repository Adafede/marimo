# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "great-tables==0.21.0",
#     "marimo",
#     "polars==1.39.3",
#     "rdflib==7.6.0",
# ]
# [tool.marimo.runtime]
# output_max_bytes = 300_000_000
# ///

"""LOTUS Wikidata Explorer.

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

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import polars as pl
    import array
    import csv
    import io
    import json
    import time
    import hashlib
    import re
    import sys
    import urllib.parse
    import gc
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from datetime import date, datetime
    from typing import Any

    # Check for WASM/Pyodide environment early
    IS_PYODIDE = "pyodide" in sys.modules

    # rdflib is always available (WASM compatible)
    from rdflib import Graph, Literal, URIRef
    from rdflib.namespace import XSD, Namespace

    # maplib is faster but not WASM compatible
    MaplibModel: Any = None
    if not IS_PYODIDE:
        try:
            from maplib import Model as MaplibModel
        except ImportError:
            pass  # Fall back to rdflib

    _USE_LOCAL = True
    if _USE_LOCAL:
        sys.path.insert(0, ".")

    from modules.text.formula.filters import FormulaFilters
    from modules.text.formula.create_filters import create_filters
    from modules.text.formula.serialize_filters import serialize_filters
    from modules.text.formula.match_filters import match_filters
    from modules.text.smiles.validate_and_escape import validate_and_escape
    from modules.text.strings.pluralize import pluralize
    from modules.knowledge.wikidata.entity.extract_from_url import extract_from_url
    from modules.knowledge.wikidata.url.constants import (
        ENTITY_PREFIX as WIKIDATA_ENTITY_PREFIX,
        STATEMENT_PREFIX as WIKIDATA_STATEMENT_PREFIX,
        WIKIDATA_HTTP_BASE,
        WIKI_PREFIX,
    )
    from modules.knowledge.wikidata.html.scholia import scholia_url
    from modules.knowledge.wikidata.sparql.query_taxon_search import query_taxon_search
    from modules.knowledge.wikidata.sparql.query_taxon_connectivity import (
        query_taxon_connectivity,
    )
    from modules.knowledge.wikidata.sparql.query_taxon_details import (
        query_taxon_details,
    )
    from modules.knowledge.wikidata.sparql.query_compounds import (
        query_compounds_by_taxon,
        query_all_compounds,
    )
    from modules.knowledge.wikidata.sparql.query_sachem import query_sachem
    from modules.net.sparql.execute_with_retry import execute_with_retry
    from modules.net.sparql.parse_response import parse_sparql_response
    from modules.net.sparql.values_clause import values_clause
    from modules.chem.cdk.depict.svg_from_smiles import svg_from_smiles
    from modules.knowledge.rdf.namespace.wikidata import WIKIDATA_NAMESPACES
    from modules.io.compress.if_large import compress_if_large

    if IS_PYODIDE:
        import pyodide_http

        pyodide_http.patch_all()

    CONFIG: dict[str, Any] = {
        "app_version": "0.1.0",
        "app_name": "LOTUS Wikidata Explorer",
        "app_url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
        "qlever_endpoint": "https://qlever.dev/api/wikidata",
        "table_row_limit": 100 if IS_PYODIDE else 1_000,
        "download_embed_threshold_bytes": 10 if IS_PYODIDE else 100,
        "color_hyperlink": "#3377c4",
        "color_wikidata_blue": "#006699",
        "color_wikidata_green": "#339966",
        "color_wikidata_red": "#990000",
        "page_size_default": 10,
        "page_size_export": 10,
    }

    PLURAL_MAP = {
        "Entry": "Entries",
        "entry": "entries",
        "Taxon": "Taxa",
        "taxon": "taxa",
    }

    YEAR = date.today().year

    # ========================================================================
    # DOMAIN MODELS
    # ========================================================================

    @dataclass(frozen=True, slots=True)
    class SearchCriteria:
        """Immutable search parameters."""

        taxon: str = ""
        smiles: str = ""
        smiles_search_type: str = "substructure"
        smiles_threshold: float = 0.8
        mass_range: tuple[float, float] = (0.0, 2000.0)
        year_range: tuple[int, int] = (1900, datetime.now().year)
        formula_filters: FormulaFilters | None = None

        def has_mass_filter(self) -> bool:
            """Return whether the mass range differs from the default bounds."""
            return self.mass_range != (0.0, 2000.0)

        def has_year_filter(self) -> bool:
            """Return whether the publication year range differs from defaults."""
            return self.year_range != (1900, datetime.now().year)

        def to_filters_dict(self) -> dict:
            """Convert to filters dictionary.

            Returns
            -------
            dict
                Dictionary containing to filters dict.

            """
            filters: dict[str, Any] = {}
            if self.smiles:
                chem_struct: dict[str, Any] = {
                    "smiles": self.smiles,
                    "search_type": self.smiles_search_type,
                }
                if self.smiles_search_type == "similarity":
                    chem_struct["similarity_threshold"] = self.smiles_threshold
                filters["chemical_structure"] = chem_struct
            if self.has_mass_filter():
                filters["mass"] = {"min": self.mass_range[0], "max": self.mass_range[1]}
            if self.has_year_filter():
                filters["publication_year"] = {
                    "start": self.year_range[0],
                    "end": self.year_range[1],
                }
            if self.formula_filters:
                formula_dict = serialize_filters(self.formula_filters)
                if formula_dict:
                    filters["molecular_formula"] = formula_dict
            return filters

    @dataclass(frozen=True, slots=True)
    class DatasetStats:
        """Dataset statistics."""

        n_compounds: int = 0
        n_taxa: int = 0
        n_references: int = 0
        n_entries: int = 0

        @classmethod
        def from_lazyframe(cls, df: pl.LazyFrame) -> "DatasetStats":
            """Compute dataset statistics from a lazy Polars frame."""
            stats: pl.DataFrame = df.select(
                [
                    pl.col("compound")
                    .approx_n_unique()
                    .cast(pl.UInt32)
                    .alias("n_compounds"),
                    pl.col("taxon").approx_n_unique().cast(pl.UInt32).alias("n_taxa"),
                    pl.col("reference")
                    .approx_n_unique()
                    .cast(pl.UInt32)
                    .alias("n_refs"),
                    pl.len().cast(pl.UInt32).alias("n_entries"),
                ],
            ).collect()

            if stats.is_empty():
                return cls(0, 0, 0, 0)

            return cls(
                n_compounds=int(stats["n_compounds"][0]),
                n_taxa=int(stats["n_taxa"][0]),
                n_references=int(stats["n_refs"][0]),
                n_entries=int(stats["n_entries"][0]),
            )

    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================

    class MemoryManager:
        """Tune memory-sensitive execution parameters for WASM and native runtimes."""

        def __init__(self, is_wasm: bool):
            """Initialize runtime memory mode.

            Parameters
            ----------
            is_wasm : bool
                Whether the app runs in a WASM environment.

            """
            self.is_wasm = is_wasm

        def get_batch_size(self, format: str) -> int:
            """Return export batch size tuned for runtime and output format."""
            sizes = {"csv": (2000, 10000), "json": (5000, 10000), "ttl": (500, 2000)}
            wasm_size, desktop_size = sizes.get(format, (1000, 5000))
            return wasm_size if self.is_wasm else desktop_size

    # ========================================================================
    # NORMALIZED DATA STORAGE (Memory-efficient for WASM)
    # ========================================================================

    @dataclass
    class NormalizedDataset:
        """Memory-efficient normalized storage for compound-taxon-reference data.

        Instead of storing denormalized rows with repeated metadata, we store:
        - facts: DataFrame with just IDs (compound, taxon, reference, statement, ref)
        - compound_meta: DataFrame with compound metadata (name, inchikey, smiles, mass, mf)
        - taxon_meta: DataFrame with taxon metadata (taxon_name)
        - ref_meta: DataFrame with reference metadata (ref_title, ref_doi, pub_date)

        This significantly reduces memory usage for large datasets where the same
        compound/taxon/reference appears in multiple rows.
        """

        facts: pl.LazyFrame  # compound, taxon, reference, statement, ref
        compound_meta: pl.LazyFrame  # compound, name, inchikey, smiles, mass, mf
        taxon_meta: pl.LazyFrame  # taxon, taxon_name
        ref_meta: pl.LazyFrame  # reference, ref_title, ref_doi, pub_date

        def to_denormalized(self) -> pl.LazyFrame:
            """Join all tables to produce the full denormalized view.

            Returns
            -------
            pl.LazyFrame
                LazyFrame containing to denormalized.

            """
            return (
                self.facts.join(self.compound_meta, on="compound", how="left")
                .join(self.taxon_meta, on="taxon", how="left")
                .join(self.ref_meta, on="reference", how="left")
            )

        @classmethod
        def empty(cls) -> "NormalizedDataset":
            """Create an empty normalized dataset.

            Returns
            -------
            'NormalizedDataset'
                'NormalizedDataset' Empty normalized dataset with predefined schemas for all tables.

            """
            return cls(
                facts=pl.LazyFrame(
                    schema={
                        "compound": pl.UInt32,
                        "taxon": pl.UInt32,
                        "reference": pl.UInt32,
                        "statement": pl.Utf8,
                        "ref": pl.Utf8,
                    },
                ),
                compound_meta=pl.LazyFrame(
                    schema={
                        "compound": pl.UInt32,
                        "name": pl.Utf8,
                        "inchikey": pl.Utf8,
                        "smiles": pl.Utf8,
                        "mass": pl.Float32,
                        "mf": pl.Utf8,
                    },
                ),
                taxon_meta=pl.LazyFrame(
                    schema={
                        "taxon": pl.UInt32,
                        "taxon_name": pl.Utf8,
                    },
                ),
                ref_meta=pl.LazyFrame(
                    schema={
                        "reference": pl.UInt32,
                        "ref_title": pl.Utf8,
                        "ref_doi": pl.Utf8,
                        "pub_date": pl.Date,
                    },
                ),
            )

        @classmethod
        def from_csv_bytes(cls, csv_bytes: bytes) -> "NormalizedDataset":
            """Parse CSV bytes in a streaming manner and build normalized tables.

                                                This is the key memory optimization: instead of loading the entire
                                                denormalized CSV into memory, we stream through it once and build
                                                deduplicated lookup tables.

                                                Uses io.TextIOWrapper for true streaming without creating a full
                                                string copy of the data.

            Parameters
            ----------
            csv_bytes : bytes
                Csv bytes.

            Returns
            -------
            'NormalizedDataset'
                'NormalizedDataset' Normalized dataset built from streamed CSV rows.

            """
            # Quick empty check without creating a copy via strip()
            if not csv_bytes or len(csv_bytes) < 10:
                return cls.empty()

            # Dictionaries for deduplication (use dict for O(1) lookup)
            compound_meta: dict[
                int,
                tuple,
            ] = {}  # compound_id -> (name, inchikey, smiles, mass, mf)
            taxon_meta: dict[int, str | None] = {}  # taxon_id -> taxon_name
            ref_meta: dict[int, tuple] = {}  # ref_id -> (ref_title, ref_doi, pub_date)

            # Lists for fact table - use array for integers (more compact)
            facts_compound: array.array = array.array("I")  # unsigned int
            facts_taxon: array.array = array.array("I")
            facts_reference: array.array = array.array("I")
            facts_statement: list[str | None] = []
            facts_ref: list[str | None] = []

            # Create BytesIO wrapper
            bytes_stream = io.BytesIO(csv_bytes)
            text_stream = io.TextIOWrapper(
                bytes_stream,
                encoding="utf-8",
                errors="replace",
            )
            reader = csv.DictReader(text_stream)

            row_count = 0
            gc_interval = 5000 if IS_PYODIDE else 50000  # More frequent GC in WASM

            for row in reader:
                row_count += 1
                # Extract IDs (these are returned as integers from SPARQL)
                try:
                    compound_id = int(row.get("compound") or 0)
                    taxon_id = int(row.get("taxon") or 0)
                    ref_id = int(row.get("ref_qid") or row.get("reference") or 0)
                except (ValueError, TypeError):
                    continue  # Skip malformed rows

                if compound_id == 0:
                    continue  # Skip rows without compound

                # Add to fact table (use array.append for compact int storage)
                facts_compound.append(compound_id)
                facts_taxon.append(taxon_id)
                facts_reference.append(ref_id)
                # Don't intern statement/ref - they're mostly unique URIs
                stmt = row.get("statement")
                ref_val = row.get("ref")
                facts_statement.append(stmt if stmt else None)
                facts_ref.append(ref_val if ref_val else None)

                # Deduplicate compound metadata
                if compound_id not in compound_meta:
                    # Prefer isomeric SMILES, fall back to connectivity SMILES
                    smiles = (
                        row.get("compound_smiles_iso")
                        or row.get("compound_smiles_conn")
                        or None
                    )
                    try:
                        _mass_val = row.get("compound_mass")
                        mass = float(_mass_val) if _mass_val else None
                    except (ValueError, TypeError):
                        mass = None
                    # Intern strings to save memory on repeated values
                    name = row.get("compoundLabel")
                    inchikey = row.get("compound_inchikey")
                    mf = row.get("compound_formula")
                    compound_meta[compound_id] = (
                        sys.intern(name) if name else None,
                        sys.intern(inchikey) if inchikey else None,
                        smiles,  # SMILES are usually unique, don't intern
                        mass,
                        sys.intern(mf) if mf else None,
                    )

                # Deduplicate taxon metadata
                if taxon_id and taxon_id not in taxon_meta:
                    taxon_name = row.get("taxon_name")
                    taxon_meta[taxon_id] = (
                        sys.intern(taxon_name) if taxon_name else None
                    )

                # Deduplicate reference metadata
                if ref_id and ref_id not in ref_meta:
                    # Extract DOI (remove URL prefix if present)
                    ref_doi = row.get("ref_doi") or None
                    if ref_doi and ref_doi.startswith("http"):
                        parts = ref_doi.split("doi.org/")
                        ref_doi = parts[-1] if len(parts) > 1 else ref_doi

                    # Parse date
                    pub_date_str = row.get("ref_date") or None
                    pub_date = None
                    if pub_date_str:
                        try:
                            pub_date = datetime.strptime(
                                pub_date_str,
                                "%Y-%m-%dT%H:%M:%SZ",
                            ).date()
                        except ValueError:
                            pass

                    ref_title = row.get("ref_title")
                    ref_meta[ref_id] = (
                        sys.intern(ref_title) if ref_title else None,
                        sys.intern(ref_doi) if ref_doi else None,
                        pub_date,
                    )

                # Periodic GC in WASM to prevent memory buildup
                if IS_PYODIDE and row_count % gc_interval == 0:
                    gc.collect()

            # Close and free the streams
            text_stream.close()
            bytes_stream.close()
            del text_stream, bytes_stream
            # Trigger GC after freeing streams - this is crucial to free csv_bytes reference
            if IS_PYODIDE:
                gc.collect()

            # Convert arrays to lists for polars (polars doesn't accept array.array directly)
            facts_compound_list = facts_compound.tolist()
            facts_taxon_list = facts_taxon.tolist()
            facts_reference_list = facts_reference.tolist()
            # Free the arrays immediately
            del facts_compound, facts_taxon, facts_reference
            if IS_PYODIDE:
                gc.collect()

            # Build DataFrames from collected data
            facts_df = pl.LazyFrame(
                {
                    "compound": facts_compound_list,
                    "taxon": facts_taxon_list,
                    "reference": facts_reference_list,
                    "statement": facts_statement,
                    "ref": facts_ref,
                },
                schema={
                    "compound": pl.UInt32,
                    "taxon": pl.UInt32,
                    "reference": pl.UInt32,
                    "statement": pl.Utf8,
                    "ref": pl.Utf8,
                },
            )

            # Clear fact lists to free memory
            del facts_compound_list, facts_taxon_list, facts_reference_list
            del facts_statement, facts_ref

            # Build compound metadata DataFrame
            compound_ids = list(compound_meta.keys())
            compound_data = list(compound_meta.values())
            compound_meta_df = pl.LazyFrame(
                {
                    "compound": compound_ids,
                    "name": [c[0] for c in compound_data],
                    "inchikey": [c[1] for c in compound_data],
                    "smiles": [c[2] for c in compound_data],
                    "mass": [c[3] for c in compound_data],
                    "mf": [c[4] for c in compound_data],
                },
                schema={
                    "compound": pl.UInt32,
                    "name": pl.Utf8,
                    "inchikey": pl.Utf8,
                    "smiles": pl.Utf8,
                    "mass": pl.Float32,
                    "mf": pl.Utf8,
                },
            )
            del compound_meta, compound_ids, compound_data

            # Build taxon metadata DataFrame
            taxon_ids = list(taxon_meta.keys())
            taxon_names = list(taxon_meta.values())
            taxon_meta_df = pl.LazyFrame(
                {
                    "taxon": taxon_ids,
                    "taxon_name": taxon_names,
                },
                schema={
                    "taxon": pl.UInt32,
                    "taxon_name": pl.Utf8,
                },
            )
            del taxon_meta, taxon_ids, taxon_names

            # Build reference metadata DataFrame
            ref_ids = list(ref_meta.keys())
            ref_data = list(ref_meta.values())
            ref_meta_df = pl.LazyFrame(
                {
                    "reference": ref_ids,
                    "ref_title": [r[0] for r in ref_data],
                    "ref_doi": [r[1] for r in ref_data],
                    "pub_date": [r[2] for r in ref_data],
                },
                schema={
                    "reference": pl.UInt32,
                    "ref_title": pl.Utf8,
                    "ref_doi": pl.Utf8,
                    "pub_date": pl.Date,
                },
            )
            del ref_meta, ref_ids, ref_data

            # Trigger garbage collection if in WASM
            if IS_PYODIDE:
                gc.collect()

            return cls(
                facts=facts_df,
                compound_meta=compound_meta_df,
                taxon_meta=taxon_meta_df,
                ref_meta=ref_meta_df,
            )

    # ========================================================================
    # SERVICES
    # ========================================================================

    class WikidataQueryService:
        """Fetch compound datasets from Wikidata and Sachem endpoints."""

        def __init__(self, endpoint: str, use_normalized: bool = False):
            """Initialize query service settings.

            Parameters
            ----------
            endpoint : str
                SPARQL endpoint URL.
            use_normalized : bool
                Whether to parse responses into normalized intermediate tables.

            """
            self.endpoint = endpoint
            self.use_normalized = use_normalized

        def fetch_compounds(
            self,
            qid: str,
            smiles: str | None = None,
            smiles_search_type: str = "substructure",
            smiles_threshold: float = 0.8,
        ) -> pl.LazyFrame:
            """Fetch compounds from Wikidata.

                                                In WASM mode with use_normalized=True, this uses the memory-efficient
                                                normalized storage that parses CSV streaming and deduplicates metadata.

            Parameters
            ----------
            qid : str
                Qid.
            smiles : str | None
                None. Default is None.
            smiles_search_type : str
                Default is 'substructure'.
            smiles_threshold : float
                Default is 0.8.

            Returns
            -------
            pl.LazyFrame
                LazyFrame containing compounds.

            """
            query = self._build_query(qid, smiles, smiles_search_type, smiles_threshold)
            csv_bytes = execute_with_retry(query, self.endpoint)

            # Quick empty check without creating a copy via strip()
            if not csv_bytes or len(csv_bytes) < 10:
                return pl.LazyFrame()

            if self.use_normalized:
                # Use memory-efficient normalized parsing
                dataset = NormalizedDataset.from_csv_bytes(csv_bytes)
                # Clear the original bytes to free memory immediately
                del csv_bytes
                if IS_PYODIDE:
                    gc.collect()
                return dataset.to_denormalized()
            else:
                # Use standard polars scan_csv
                return pl.scan_csv(
                    io.BytesIO(csv_bytes),
                    low_memory=True,
                    rechunk=False,
                    schema_overrides={
                        "compound": pl.UInt32,
                        "taxon": pl.UInt32,
                        "reference": pl.UInt32,
                        "compound_mass": pl.Float32,
                        "name": pl.Utf8,
                        "inchikey": pl.Utf8,
                        "smiles": pl.Utf8,
                        "taxon_name": pl.Utf8,
                        "ref_title": pl.Utf8,
                        "ref_doi": pl.Utf8,
                        "mf": pl.Utf8,
                        "statement": pl.Utf8,
                        "ref": pl.Utf8,
                    },
                )

        def _build_query(
            self,
            qid: str,
            smiles: str | None,
            search_type: str,
            threshold: float,
        ) -> str:
            if smiles:
                # Molfile-like multiline queries are handled in substructure mode.
                effective_search_type = search_type
                if ("\n" in smiles or "\r" in smiles) and search_type == "similarity":
                    effective_search_type = "substructure"

                if not qid or qid == "":
                    effective_qid = None
                elif qid == "*":
                    effective_qid = "Q2382443"  # Biota
                else:
                    effective_qid = qid

                return query_sachem(
                    escaped_smiles=validate_and_escape(smiles) or "",
                    search_type=effective_search_type,
                    threshold=threshold,
                    taxon_qid=effective_qid,
                )
            elif qid == "*":
                return query_all_compounds()
            else:
                return query_compounds_by_taxon(qid)

    class DataTransformService:
        """Apply canonical post-processing steps to query result tables."""

        @staticmethod
        def apply_standard_transforms(
            df: pl.LazyFrame,
            from_normalized: bool = False,
        ) -> pl.LazyFrame:
            """Apply standard transformations to the data.

            Parameters
            ----------
            df : pl.LazyFrame
                Df.
            from_normalized : bool
                False. Default is False.

            Returns
            -------
            pl.LazyFrame
                LazyFrame containing apply standard transforms.

            """
            if from_normalized:
                # Normalized data already has correct column names and types
                df = DataTransformService.add_missing_columns(df)
                df = DataTransformService.deduplicate(df)
            else:
                # Full transformation pipeline for raw CSV data
                df = DataTransformService.rename_columns(df)
                df = DataTransformService.combine_smiles(df)
                df = DataTransformService.extract_doi(df)
                df = DataTransformService.parse_dates(df)
                df = DataTransformService.cast_types(df)
                df = DataTransformService.drop_old_columns(df)
                df = DataTransformService.add_missing_columns(df)
                df = DataTransformService.deduplicate(df)
            return df

        @staticmethod
        def rename_columns(df: pl.LazyFrame) -> pl.LazyFrame:
            """Rename raw query columns to canonical internal names."""
            return df.rename(
                {
                    "compoundLabel": "name",
                    "compound_inchikey": "inchikey",
                    "ref_qid": "reference",
                    "ref_date": "pub_date",
                    "compound_mass": "mass",
                    "compound_formula": "mf",
                },
            )

        @staticmethod
        def combine_smiles(df: pl.LazyFrame) -> pl.LazyFrame:
            """Combine isomeric and connectivity SMILES into a single column."""
            return df.with_columns(
                [
                    pl.coalesce(["compound_smiles_iso", "compound_smiles_conn"]).alias(
                        "smiles",
                    ),
                ],
            )

        @staticmethod
        def extract_doi(df: pl.LazyFrame) -> pl.LazyFrame:
            """Normalize DOI values by removing URL prefixes when present."""
            return df.with_columns(
                [
                    pl.when(pl.col("ref_doi").cast(pl.Utf8).str.starts_with("http"))
                    .then(
                        pl.col("ref_doi")
                        .cast(pl.Utf8)
                        .str.split("doi.org/")
                        .list.last(),
                    )
                    .otherwise(pl.col("ref_doi"))
                    .alias("ref_doi"),
                ],
            )

        @staticmethod
        def parse_dates(df: pl.LazyFrame) -> pl.LazyFrame:
            """Parse publication date strings into date values."""
            return df.with_columns(
                [
                    pl.when(
                        pl.col("pub_date").is_not_null() & (pl.col("pub_date") != ""),
                    )
                    .then(
                        pl.col("pub_date")
                        .str.strptime(
                            pl.Datetime,
                            format="%Y-%m-%dT%H:%M:%SZ",
                            strict=False,
                        )
                        .dt.date(),
                    )
                    .otherwise(None)
                    .alias("pub_date"),
                ],
            )

        @staticmethod
        def cast_types(df: pl.LazyFrame) -> pl.LazyFrame:
            """Cast key columns to stable numeric and float dtypes."""
            return df.with_columns(
                [
                    pl.col("compound").cast(pl.UInt32),
                    # pl.col("name").cast(pl.Utf8),
                    # pl.col("inchikey").cast(pl.Utf8),
                    # pl.col("smiles").cast(pl.Utf8),
                    # pl.col("taxon_name").cast(pl.Utf8),
                    pl.col("taxon").cast(pl.UInt32),
                    # pl.col("ref_title").cast(pl.Utf8),
                    # pl.col("ref_doi").cast(pl.Utf8),
                    pl.col("reference").cast(pl.UInt32),
                    pl.col("mass").cast(pl.Float32),
                    # pl.col("mf").cast(pl.Utf8),
                    # pl.col("statement").cast(pl.Utf8),
                    # pl.col("ref").cast(pl.Utf8),
                ],
            )

        @staticmethod
        def drop_old_columns(df: pl.LazyFrame) -> pl.LazyFrame:
            """Drop intermediate raw columns no longer needed downstream."""
            to_drop = ["compound_smiles_iso", "compound_smiles_conn"]
            return df.drop(
                [col for col in to_drop if col in df.collect_schema().names()],
            )

        @staticmethod
        def add_missing_columns(df: pl.LazyFrame) -> pl.LazyFrame:
            """Add required columns that are absent from the current schema."""
            required = [
                "compound",
                "name",
                "inchikey",
                "smiles",
                "taxon_name",
                "taxon",
                "ref_title",
                "ref_doi",
                "reference",
                "pub_date",
                "mass",
                "mf",
                "statement",
                "ref",
            ]
            missing = [
                col for col in required if col not in df.collect_schema().names()
            ]
            if missing:
                df = df.with_columns([pl.lit(None).alias(col) for col in missing])
            return df

        @staticmethod
        def deduplicate(df: pl.LazyFrame) -> pl.LazyFrame:
            """Deduplicate rows by compound, taxon, and reference identifiers."""
            return df.unique(
                subset=["compound", "taxon", "reference"],
                keep="first",
            )

    class FilterService:
        """Apply user-selected numeric and formula filters to datasets."""

        @staticmethod
        def apply_filters(df: pl.LazyFrame, criteria: SearchCriteria) -> pl.LazyFrame:
            """Apply all active search criteria to the dataset."""
            if criteria.has_year_filter():
                df = FilterService.filter_by_year(df, criteria.year_range)
            if criteria.has_mass_filter():
                df = FilterService.filter_by_mass(df, criteria.mass_range)
            if criteria.formula_filters and criteria.formula_filters.is_active():
                df = FilterService.filter_by_formula(df, criteria.formula_filters)
            return df

        @staticmethod
        def filter_by_year(
            df: pl.LazyFrame,
            year_range: tuple[int, int],
        ) -> pl.LazyFrame:
            """Filter rows by inclusive publication year bounds."""
            year_start, year_end = year_range
            if year_start:
                df = df.filter(pl.col("pub_date").dt.year() >= year_start)
            if year_end:
                df = df.filter(pl.col("pub_date").dt.year() <= year_end)
            return df

        @staticmethod
        def filter_by_mass(
            df: pl.LazyFrame,
            mass_range: tuple[float, float],
        ) -> pl.LazyFrame:
            """Filter rows by inclusive exact-mass bounds."""
            mass_min, mass_max = mass_range
            if mass_min:
                df = df.filter(pl.col("mass") >= mass_min)
            if mass_max:
                df = df.filter(pl.col("mass") <= mass_max)
            return df

        @staticmethod
        def filter_by_formula(
            df: pl.LazyFrame,
            formula_filters: FormulaFilters,
        ) -> pl.LazyFrame:
            """Filter rows by molecular formula criteria."""
            return df.filter(
                pl.col("mf").map_batches(
                    lambda s: s.map_elements(
                        lambda f: match_filters(f or "", formula_filters),
                        return_dtype=pl.Boolean,
                    ),
                ),
            )

    class TaxonResolutionService:
        """Resolve user taxon input to a single Wikidata taxon QID."""

        def __init__(self, endpoint: str):
            """Initialize taxon resolution service.

            Parameters
            ----------
            endpoint : str
                SPARQL endpoint URL used for taxon lookup.

            """
            self.endpoint = endpoint

        def resolve(self, taxon_input: str) -> tuple[str | None, mo.Html | None]:
            """Resolve free-text or QID taxon input to a target QID."""
            taxon_input = str(taxon_input).strip()

            if not taxon_input:
                return None, None
            if taxon_input == "*":
                return "*", None
            if taxon_input.upper().startswith("Q") and taxon_input[1:].isdigit():
                return taxon_input.upper(), None

            # Sanitize multi-part taxon names
            original_input = taxon_input
            if " " in taxon_input or "_" in taxon_input:
                parts = taxon_input.replace("_", " ").split()
                taxon_input = parts[0].capitalize() + " " + " ".join(parts[1:])

            try:
                query = query_taxon_search(taxon_input)
                csv_bytes = execute_with_retry(
                    query,
                    self.endpoint,
                    fallback_endpoint=None,
                )

                if not csv_bytes or not csv_bytes.strip():
                    return None, None

                df: pl.DataFrame = parse_sparql_response(csv_bytes).collect()
                matches = [
                    (
                        extract_from_url(row["taxon"], WIKIDATA_ENTITY_PREFIX),
                        row["taxon_name"],
                    )
                    for row in df.iter_rows(named=True)
                    if row.get("taxon") and row.get("taxon_name")
                ]

                if not matches:
                    return None, None

                taxon_lower = taxon_input.lower()
                exact_matches = [
                    (qid, name) for qid, name in matches if name.lower() == taxon_lower
                ]

                if len(exact_matches) == 1:
                    # Single exact match - show sanitization notice if input was modified
                    if original_input != taxon_input:
                        notice_html = self._create_sanitization_notice(
                            original_input,
                            taxon_input,
                        )
                        return exact_matches[0][0], notice_html
                    return exact_matches[0][0], None

                if len(exact_matches) > 1:
                    return self._resolve_ambiguous(
                        exact_matches,
                        is_exact=True,
                        original_input=original_input,
                        sanitized_input=taxon_input,
                    )

                if len(matches) > 1:
                    return self._resolve_ambiguous(
                        matches,
                        is_exact=False,
                        original_input=original_input,
                        sanitized_input=taxon_input,
                    )

                # Single non-exact match - show sanitization notice if input was modified
                if original_input != taxon_input:
                    notice_html = self._create_sanitization_notice(
                        original_input,
                        taxon_input,
                    )
                    return matches[0][0], notice_html
                return matches[0][0], None

            except (ConnectionError, RuntimeError, TimeoutError):
                # Always propagate HTTP/network/timeout errors so the caller
                # can surface them to the user, regardless of how Pyodide
                # wraps the underlying JS fetch error.
                raise
            except Exception:
                return None, None

        @staticmethod
        def _is_server_error(exc: ConnectionError) -> bool:
            """Return True for HTTP 5xx errors surfaced by execute_with_retry.

            Parameters
            ----------
            exc : ConnectionError
                Exc.

            Returns
            -------
            bool
                Result is server error.

            """
            status = TaxonResolutionService._extract_http_status(exc)
            if status is None:
                return False
            return 500 <= status <= 599

        @staticmethod
        def _extract_http_status(exc: Exception) -> int | None:
            """Extract an HTTP status code from nested exception metadata."""
            for source in (exc, getattr(exc, "__cause__", None)):
                if source is None:
                    continue
                for attr in ("code", "status"):
                    value = getattr(source, attr, None)
                    if isinstance(value, int) and 100 <= value <= 599:
                        return value

            text = str(exc)
            cause = getattr(exc, "__cause__", None)
            if cause is not None:
                text = f"{text} {cause}"

            for pattern in (
                r"HTTP\s+status\s*(\d{3})",
                r"HTTP\s+Error\s*(\d{3})",
                r"status[=: ]+\s*(\d{3})",
            ):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            return None

        def _create_sanitization_notice(
            self,
            original_input: str,
            sanitized_input: str,
        ) -> mo.Html:
            """Create HTML notice for sanitized input.

            Parameters
            ----------
            original_input : str
                Original input.
            sanitized_input : str
                Sanitized input.

            Returns
            -------
            mo.Html
                Result create sanitization notice.

            """
            html = f"""
            <div style="line-height: 1.6; color: {CONFIG["color_hyperlink"]};">
                Input standardized from "<strong>{original_input}</strong>" to "<strong>{sanitized_input}</strong>"
            </div>
            """
            return mo.Html(html)

        def _resolve_ambiguous(
            self,
            matches: list[tuple[str | None, str]],
            is_exact: bool,
            original_input: str = "",
            sanitized_input: str = "",
        ) -> tuple[str, mo.Html]:
            """Resolve ambiguous taxon matches.

            Parameters
            ----------
            matches : list[tuple[str | None, str]]
                Matches.
            is_exact : bool
                Is exact.
            original_input : str
                Default is ''.
            sanitized_input : str
                Default is ''.

            Returns
            -------
            tuple[str, mo.Html]
                Tuple containing resolve ambiguous.

            """
            qids = tuple(qid for qid, _ in matches if qid is not None)
            info = {qid: [0, "", "", ""] for qid in qids}

            csv_bytes = execute_with_retry(
                query_taxon_connectivity(values_clause("taxon", qids, prefix="wd:")),
                endpoint=self.endpoint,
                fallback_endpoint=None,
            )
            if csv_bytes and csv_bytes.strip():
                for row in (
                    parse_sparql_response(csv_bytes).collect().iter_rows(named=True)
                ):
                    taxon_url = row.get("taxon")
                    if taxon_url:
                        qid = extract_from_url(taxon_url, WIKIDATA_ENTITY_PREFIX)
                        if qid is not None:
                            info[qid][0] = int(row.get("count") or 0)

            csv_bytes = execute_with_retry(
                query_taxon_details(values_clause("taxon", qids, prefix="wd:")),
                endpoint=self.endpoint,
                fallback_endpoint=None,
            )
            if csv_bytes and csv_bytes.strip():
                for row in (
                    parse_sparql_response(csv_bytes).collect().iter_rows(named=True)
                ):
                    taxon_url = row.get("taxon")
                    if taxon_url:
                        qid = extract_from_url(taxon_url, WIKIDATA_ENTITY_PREFIX)
                        if qid is not None:
                            info[qid][1] = row.get("taxonDescription", "")
                            info[qid][2] = row.get("taxon_parentLabel", "")

            selected_qid = max(qids, key=lambda q: info[q][0])

            matches_sorted = sorted(matches, key=lambda x: info[x[0]][0], reverse=True)
            matches_with_details = [
                (qid, name, info[qid][1], info[qid][2], info[qid][0])
                for qid, name in matches_sorted
                if qid is not None
            ]

            return selected_qid, self._create_taxon_warning_html(
                matches_with_details,
                selected_qid,
                is_exact,
                original_input,
                sanitized_input,
            )

        def _create_taxon_warning_html(
            self,
            matches: list,
            selected_qid: str,
            is_exact: bool,
            original_input: str = "",
            sanitized_input: str = "",
        ) -> mo.Html:
            """Create HTML warning for ambiguous taxon.

            Parameters
            ----------
            matches : list
                Matches.
            selected_qid : str
                Selected qid.
            is_exact : bool
                Is exact.
            original_input : str
                Default is ''.
            sanitized_input : str
                Default is ''.

            Returns
            -------
            mo.Html
                Result create taxon warning html.

            """
            match_type = "exact matches" if is_exact else "similar taxa"
            intro = (
                f"Ambiguous taxon name. Multiple {match_type} found:"
                if is_exact
                else "No exact match. Similar taxa found:"
            )

            # Add sanitization notice if input was modified
            sanitization_notice = ""
            if original_input and sanitized_input and original_input != sanitized_input:
                sanitization_notice = f'<div style="margin-bottom: 0.5em; color: {CONFIG["color_hyperlink"]};">Input standardized from "<strong>{original_input}</strong>" to "<strong>{sanitized_input}</strong>"</div>'

            items = []
            for match_data in matches:
                qid, name, description, parent, edges_count = match_data
                link_html = f'<a href="{scholia_url(qid)}" target="_blank" style="color: {CONFIG["color_hyperlink"]}; font-weight: bold;">{qid}</a>'

                details = []
                if name:
                    details.append(f"<em>{name}</em>")
                if description:
                    details.append(f"{description}")
                if parent:
                    details.append(f"parent: {parent}")
                if edges_count is not None:
                    details.append(f"<strong>{edges_count:,} edges</strong>")

                details_str = " - ".join(details) if details else ""

                if qid == selected_qid:
                    items.append(
                        f"<li>{link_html} {details_str} <strong>&lt; USING THIS ONE (most edges)</strong></li>",
                    )
                else:
                    items.append(f"<li>{link_html} {details_str}</li>")

            items_html = "".join(items)

            html = f"""
            <div style="line-height: 1.6;">
                {sanitization_notice}
                {intro}
                <ul style="margin: 0.5em 0; padding-left: 1.5em;">
                    {items_html}
                </ul>
                <em>For precision, use a specific QID in the search box.</em>
            </div>
            """

            return mo.Html(html)

    # ========================================================================
    # EXPORT STRATEGIES
    # ========================================================================

    class ExportStrategy(ABC):
        """Abstract base class for dataset export serializers."""

        def __init__(self, memory: MemoryManager):
            """Initialize export strategy.

            Parameters
            ----------
            memory : MemoryManager
                Memory manager used to adapt export behavior.

            """
            self.memory = memory

        def export(self, df: pl.LazyFrame) -> bytes:
            """Serialize a lazy dataframe to bytes with the concrete strategy."""
            return self._to_bytes(df)

        @abstractmethod
        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            """Serialize a lazy dataframe to bytes."""
            pass

    class CSVExportStrategy(ExportStrategy):
        """Export datasets as CSV bytes."""

        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            if self.memory.is_wasm:
                # WASM: collect and write to buffer (no file system)
                df_collected: pl.DataFrame = df.collect()
                buffer = io.BytesIO()
                df_collected.write_csv(buffer)
                result = buffer.getvalue()
                del df_collected, buffer
                gc.collect()
                return result
            else:
                # Native: use sink_csv for streaming (memory efficient)
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                    temp_path = f.name

                try:
                    df.sink_csv(temp_path)
                    with open(temp_path, "rb") as f:
                        result = f.read()
                finally:
                    os.unlink(temp_path)
                return result

    class JSONExportStrategy(ExportStrategy):
        """Export datasets as NDJSON bytes."""

        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            if self.memory.is_wasm:
                # WASM: collect and write NDJSON to buffer
                df_collected: pl.DataFrame = df.collect()
                buffer = io.BytesIO()
                df_collected.write_ndjson(buffer)
                result = buffer.getvalue()
                del df_collected, buffer
                gc.collect()
                return result
            else:
                # Native: use sink_ndjson for streaming
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(delete=False, suffix=".ndjson") as f:
                    temp_path = f.name

                try:
                    df.sink_ndjson(temp_path)
                    with open(temp_path, "rb") as f:
                        result = f.read()
                finally:
                    os.unlink(temp_path)
                return result

    class TTLExportStrategy(ExportStrategy):
        """TTL export using maplib (native, fast) or rdflib (WASM compatible)."""

        def __init__(
            self,
            memory: MemoryManager,
            taxon_input: str,
            qid: str,
            filters: dict,
        ):
            """Initialize TTL export strategy context.

            Parameters
            ----------
            memory : MemoryManager
                Memory manager used to choose backend implementation.
            taxon_input : str
                User-provided taxon input string.
            qid : str
                Resolved Wikidata QID for the selected taxon scope.
            filters : dict
                Active search filter values used in export metadata.

            """
            super().__init__(memory)
            self.taxon_input = taxon_input
            self.qid = qid
            self.filters = filters
            # Use maplib if available and not in WASM
            self._use_maplib = MaplibModel is not None and not self.memory.is_wasm

        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            if self._use_maplib:
                return self._to_bytes_maplib(df)
            else:
                return self._to_bytes_rdflib(df)

        def _to_bytes_maplib(self, df: pl.LazyFrame) -> bytes:
            """Fast export using maplib (native only).

            Parameters
            ----------
            df : pl.LazyFrame
                Df.

            Returns
            -------
            bytes
                Result to bytes maplib.

            """
            df_collected: pl.DataFrame = df.collect()

            dataset_uri, query_hash, result_hash = self._create_dataset_uri(
                df_collected,
            )

            # Create maplib model
            model = MaplibModel()
            model.add_prefixes(WIKIDATA_NAMESPACES)

            # Add metadata
            self._add_metadata_maplib(
                model,
                dataset_uri,
                len(df_collected),
                query_hash,
                result_hash,
            )

            # Build triples using vectorized Polars
            iri_df, literal_df = self._build_triples_vectorized(
                df_collected,
                dataset_uri,
            )
            del df_collected

            # Add to model (maplib accepts DataFrames directly)
            if iri_df is not None and len(iri_df) > 0:
                model.map_triples(iri_df)
                del iri_df

            if literal_df is not None and len(literal_df) > 0:
                model.map_triples(literal_df, validate_iris=False)
                del literal_df

            result = model.writes(format="turtle").encode("utf-8")
            del model
            return result

        def _to_bytes_rdflib(self, df: pl.LazyFrame) -> bytes:
            """WASM-compatible export using rdflib.

            Parameters
            ----------
            df : pl.LazyFrame
                Df.

            Returns
            -------
            bytes
                Result to bytes rdflib.

            """
            df_collected: pl.DataFrame = df.collect()

            dataset_uri, query_hash, result_hash = self._create_dataset_uri(
                df_collected,
            )

            # Create rdflib Graph
            g = Graph()
            for prefix, ns_str in WIKIDATA_NAMESPACES.items():
                g.bind(prefix, Namespace(ns_str))

            # Add metadata
            self._add_metadata_rdflib(
                g,
                dataset_uri,
                len(df_collected),
                query_hash,
                result_hash,
            )

            # Build triples using vectorized Polars
            iri_df, literal_df = self._build_triples_vectorized(
                df_collected,
                dataset_uri,
            )

            del df_collected
            if self.memory.is_wasm:
                gc.collect()

            # Add triples to graph (must iterate for rdflib)
            if iri_df is not None and len(iri_df) > 0:
                for s, p, o in iri_df.iter_rows():
                    g.add((URIRef(s), URIRef(p), URIRef(o)))
                del iri_df

            if literal_df is not None and len(literal_df) > 0:
                for s, p, o in literal_df.iter_rows():
                    g.add((URIRef(s), URIRef(p), Literal(o)))
                del literal_df

            if self.memory.is_wasm:
                gc.collect()

            result = g.serialize(format="turtle").encode("utf-8")
            del g
            if self.memory.is_wasm:
                gc.collect()
            return result

        def _create_dataset_uri(self, df: pl.DataFrame) -> tuple[str, str, str]:
            query_components = [self.qid or "", self.taxon_input or ""]
            if self.filters:
                query_components.append(json.dumps(self.filters, sort_keys=True))
            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8"),
            ).hexdigest()

            compound_qids = df.select(
                pl.col("compound_qid").drop_nulls().unique().sort(),
            ).to_series()
            result_hash = hashlib.sha256(
                "|".join(compound_qids.cast(pl.Utf8).to_list()).encode("utf-8"),
            ).hexdigest()

            return f"urn:hash:sha256:{result_hash}", query_hash, result_hash

        def _add_metadata_maplib(
            self,
            model,  # MaplibModel
            dataset_uri: str,
            record_count: int,
            query_hash: str,
            result_hash: str,
        ):
            schema_ns = WIKIDATA_NAMESPACES["schema"]
            wd_ns = WIKIDATA_NAMESPACES["wd"]
            dcterms_ns = WIKIDATA_NAMESPACES["dcterms"]
            rdf_ns = WIKIDATA_NAMESPACES["rdf"]

            if not self.taxon_input or self.taxon_input.strip() == "":
                dataset_name = "LOTUS Data - any taxon (structure-only search)"
                taxon_description = "any taxon"
            elif self.taxon_input == "*":
                dataset_name = "LOTUS Data - all taxa"
                taxon_description = "all taxa (Biota)"
            else:
                dataset_name = f"LOTUS Data - {self.taxon_input}"
                taxon_description = self.taxon_input

            iri_triples = [
                (dataset_uri, f"{rdf_ns}type", f"{schema_ns}Dataset"),
                (dataset_uri, f"{schema_ns}provider", CONFIG["app_url"]),
                (dataset_uri, f"{dcterms_ns}source", WIKIDATA_HTTP_BASE),
                (dataset_uri, f"{schema_ns}isBasedOn", f"{WIKI_PREFIX}Q104225190"),
            ]
            if self.qid and self.qid != "*" and self.qid.strip():
                iri_triples.append(
                    (dataset_uri, f"{schema_ns}about", f"{wd_ns}{self.qid}"),
                )
            elif self.taxon_input == "*":
                iri_triples.append(
                    (dataset_uri, f"{schema_ns}about", f"{wd_ns}Q2382443"),
                )

            model.map_triples(
                pl.DataFrame(
                    {
                        "subject": [t[0] for t in iri_triples],
                        "predicate": [t[1] for t in iri_triples],
                        "object": [t[2] for t in iri_triples],
                    },
                ),
            )

            literal_triples = [
                (dataset_uri, f"{schema_ns}name", dataset_name),
                (
                    dataset_uri,
                    f"{schema_ns}description",
                    f"Chemical compounds from {taxon_description}",
                ),
                (dataset_uri, f"{schema_ns}numberOfRecords", str(record_count)),
                (dataset_uri, f"{schema_ns}version", CONFIG["app_version"]),
                (dataset_uri, f"{dcterms_ns}provenance", f"Query hash: {query_hash}"),
                (dataset_uri, f"{dcterms_ns}identifier", f"sha256:{result_hash}"),
            ]
            if self.filters:
                literal_triples.append(
                    (
                        dataset_uri,
                        f"{dcterms_ns}provenance",
                        f"Search parameters: {json.dumps(self.filters, sort_keys=True)}",
                    ),
                )

            model.map_triples(
                pl.DataFrame(
                    {
                        "subject": [t[0] for t in literal_triples],
                        "predicate": [t[1] for t in literal_triples],
                        "object": [t[2] for t in literal_triples],
                    },
                ),
                validate_iris=False,
            )

        def _add_metadata_rdflib(
            self,
            g: Graph,
            dataset_uri: str,
            record_count: int,
            query_hash: str,
            result_hash: str,
        ):
            schema_ns = WIKIDATA_NAMESPACES["schema"]
            wd_ns = WIKIDATA_NAMESPACES["wd"]
            dcterms_ns = WIKIDATA_NAMESPACES["dcterms"]
            rdf_ns = WIKIDATA_NAMESPACES["rdf"]

            if not self.taxon_input or self.taxon_input.strip() == "":
                dataset_name = "LOTUS Data - any taxon (structure-only search)"
                taxon_description = "any taxon"
            elif self.taxon_input == "*":
                dataset_name = "LOTUS Data - all taxa"
                taxon_description = "all taxa (Biota)"
            else:
                dataset_name = f"LOTUS Data - {self.taxon_input}"
                taxon_description = self.taxon_input

            dataset_ref = URIRef(dataset_uri)

            # IRI triples
            g.add((dataset_ref, URIRef(f"{rdf_ns}type"), URIRef(f"{schema_ns}Dataset")))
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{schema_ns}provider"),
                    URIRef(CONFIG["app_url"]),
                ),
            )
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{dcterms_ns}source"),
                    URIRef(WIKIDATA_HTTP_BASE),
                ),
            )
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{schema_ns}isBasedOn"),
                    URIRef(f"{WIKI_PREFIX}Q104225190"),
                ),
            )

            if self.qid and self.qid != "*" and self.qid.strip():
                g.add(
                    (
                        dataset_ref,
                        URIRef(f"{schema_ns}about"),
                        URIRef(f"{wd_ns}{self.qid}"),
                    ),
                )
            elif self.taxon_input == "*":
                g.add(
                    (
                        dataset_ref,
                        URIRef(f"{schema_ns}about"),
                        URIRef(f"{wd_ns}Q2382443"),
                    ),
                )

            # Literal triples
            g.add((dataset_ref, URIRef(f"{schema_ns}name"), Literal(dataset_name)))
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{schema_ns}description"),
                    Literal(f"Chemical compounds from {taxon_description}"),
                ),
            )
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{schema_ns}numberOfRecords"),
                    Literal(record_count, datatype=XSD.integer),
                ),
            )
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{schema_ns}version"),
                    Literal(CONFIG["app_version"]),
                ),
            )
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{dcterms_ns}provenance"),
                    Literal(f"Query hash: {query_hash}"),
                ),
            )
            g.add(
                (
                    dataset_ref,
                    URIRef(f"{dcterms_ns}identifier"),
                    Literal(f"sha256:{result_hash}"),
                ),
            )

            if self.filters:
                g.add(
                    (
                        dataset_ref,
                        URIRef(f"{dcterms_ns}provenance"),
                        Literal(
                            f"Search parameters: {json.dumps(self.filters, sort_keys=True)}",
                        ),
                    ),
                )

        def _build_triples_vectorized(
            self,
            df: pl.DataFrame,
            dataset_uri: str,
        ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
            """Build all triples using vectorized Polars operations.

            Parameters
            ----------
            df : pl.DataFrame
                Df.
            dataset_uri : str
                Dataset uri.

            Returns
            -------
            tuple[pl.DataFrame | None, pl.DataFrame | None]
                Tuple containing build triples vectorized.

            """
            wd_ns = WIKIDATA_NAMESPACES["wd"]
            wdt_ns = WIKIDATA_NAMESPACES["wdt"]
            wds_ns = WIKIDATA_NAMESPACES["wds"]
            p_ns = WIKIDATA_NAMESPACES["p"]
            ps_ns = WIKIDATA_NAMESPACES["ps"]
            pr_ns = WIKIDATA_NAMESPACES["pr"]
            prov_ns = WIKIDATA_NAMESPACES["prov"]
            schema_ns = WIKIDATA_NAMESPACES["schema"]
            rdfs_ns = WIKIDATA_NAMESPACES["rdfs"]

            # Filter to rows with compound_qid
            df = df.filter(pl.col("compound_qid").is_not_null())
            if len(df) == 0:
                return None, None

            # Pre-compute URI columns (vectorized string concatenation)
            df = df.with_columns(
                [
                    (pl.lit(wd_ns) + pl.col("compound_qid")).alias("_compound_uri"),
                    pl.when(pl.col("taxon_qid").is_not_null())
                    .then(pl.lit(wd_ns) + pl.col("taxon_qid"))
                    .otherwise(pl.lit(None))
                    .alias("_taxon_uri"),
                    pl.when(pl.col("statement_id").is_not_null())
                    .then(pl.lit(wds_ns) + pl.col("statement_id"))
                    .otherwise(pl.lit("urn:bnode:stmt_") + pl.col("compound_qid"))
                    .alias("_statement_uri"),
                    pl.when(pl.col("reference_qid").is_not_null())
                    .then(pl.lit(wd_ns) + pl.col("reference_qid"))
                    .otherwise(pl.lit(None))
                    .alias("_ref_uri"),
                    pl.when(pl.col("ref").is_not_null())
                    .then(pl.col("ref"))
                    .otherwise(pl.lit("urn:bnode:ref_") + pl.col("compound_qid"))
                    .alias("_ref_node_uri"),
                ],
            )

            iri_dfs = []
            literal_dfs = []

            # 1. Dataset hasPart compound (all rows)
            iri_dfs.append(
                df.select(
                    [
                        pl.lit(dataset_uri).alias("subject"),
                        pl.lit(f"{schema_ns}hasPart").alias("predicate"),
                        pl.col("_compound_uri").alias("object"),
                    ],
                ),
            )

            # 2. Compound -> Taxon IRI triples (where taxon exists)
            taxon_df = df.filter(pl.col("taxon_qid").is_not_null())
            if len(taxon_df) > 0:
                iri_dfs.append(
                    taxon_df.select(
                        [
                            pl.col("_compound_uri").alias("subject"),
                            pl.lit(f"{p_ns}P703").alias("predicate"),
                            pl.col("_statement_uri").alias("object"),
                        ],
                    ),
                )
                iri_dfs.append(
                    taxon_df.select(
                        [
                            pl.col("_statement_uri").alias("subject"),
                            pl.lit(f"{ps_ns}P703").alias("predicate"),
                            pl.col("_taxon_uri").alias("object"),
                        ],
                    ),
                )
                iri_dfs.append(
                    taxon_df.select(
                        [
                            pl.col("_compound_uri").alias("subject"),
                            pl.lit(f"{wdt_ns}P703").alias("predicate"),
                            pl.col("_taxon_uri").alias("object"),
                        ],
                    ),
                )

            # 3. Reference IRI triples (where ref exists)
            ref_df = df.filter(pl.col("reference_qid").is_not_null())
            if len(ref_df) > 0:
                iri_dfs.append(
                    ref_df.select(
                        [
                            pl.col("_statement_uri").alias("subject"),
                            pl.lit(f"{prov_ns}wasDerivedFrom").alias("predicate"),
                            pl.col("_ref_node_uri").alias("object"),
                        ],
                    ),
                )
                iri_dfs.append(
                    ref_df.select(
                        [
                            pl.col("_ref_node_uri").alias("subject"),
                            pl.lit(f"{pr_ns}P248").alias("predicate"),
                            pl.col("_ref_uri").alias("object"),
                        ],
                    ),
                )

            # 4. Compound literal triples
            for col, pred in [
                ("compound_inchikey", f"{wdt_ns}P235"),
                ("compound_smiles", f"{wdt_ns}P233"),
                ("molecular_formula", f"{wdt_ns}P274"),
                ("compound_name", f"{rdfs_ns}label"),
            ]:
                if col in df.columns:
                    filtered = df.filter(pl.col(col).is_not_null())
                    if len(filtered) > 0:
                        literal_dfs.append(
                            filtered.select(
                                [
                                    pl.col("_compound_uri").alias("subject"),
                                    pl.lit(pred).alias("predicate"),
                                    pl.col(col).cast(pl.Utf8).alias("object"),
                                ],
                            ),
                        )

            # compound_mass as string
            if "compound_mass" in df.columns:
                mass_df = df.filter(pl.col("compound_mass").is_not_null())
                if len(mass_df) > 0:
                    literal_dfs.append(
                        mass_df.select(
                            [
                                pl.col("_compound_uri").alias("subject"),
                                pl.lit(f"{wdt_ns}P2067").alias("predicate"),
                                pl.col("compound_mass").cast(pl.Utf8).alias("object"),
                            ],
                        ),
                    )

            # 5. Taxon literals (deduplicated)
            if "taxon_name" in df.columns:
                taxon_unique = (
                    df.filter(
                        pl.col("taxon_qid").is_not_null()
                        & pl.col("taxon_name").is_not_null(),
                    )
                    .unique(subset=["taxon_qid"])
                    .select(["_taxon_uri", "taxon_name"])
                )
                if len(taxon_unique) > 0:
                    literal_dfs.append(
                        taxon_unique.select(
                            [
                                pl.col("_taxon_uri").alias("subject"),
                                pl.lit(f"{wdt_ns}P225").alias("predicate"),
                                pl.col("taxon_name").alias("object"),
                            ],
                        ),
                    )
                    literal_dfs.append(
                        taxon_unique.select(
                            [
                                pl.col("_taxon_uri").alias("subject"),
                                pl.lit(f"{rdfs_ns}label").alias("predicate"),
                                pl.col("taxon_name").alias("object"),
                            ],
                        ),
                    )

            # 6. Reference literals (deduplicated)
            ref_unique = df.filter(pl.col("reference_qid").is_not_null()).unique(
                subset=["reference_qid"],
            )
            if len(ref_unique) > 0:
                for col, pred in [
                    ("reference_title", f"{wdt_ns}P1476"),
                    ("reference_doi", f"{wdt_ns}P356"),
                ]:
                    if col in ref_unique.columns:
                        filtered = ref_unique.filter(pl.col(col).is_not_null())
                        if len(filtered) > 0:
                            literal_dfs.append(
                                filtered.select(
                                    [
                                        pl.col("_ref_uri").alias("subject"),
                                        pl.lit(pred).alias("predicate"),
                                        pl.col(col).cast(pl.Utf8).alias("object"),
                                    ],
                                ),
                            )

                # rdfs:label for reference_title
                if "reference_title" in ref_unique.columns:
                    filtered = ref_unique.filter(
                        pl.col("reference_title").is_not_null(),
                    )
                    if len(filtered) > 0:
                        literal_dfs.append(
                            filtered.select(
                                [
                                    pl.col("_ref_uri").alias("subject"),
                                    pl.lit(f"{rdfs_ns}label").alias("predicate"),
                                    pl.col("reference_title")
                                    .cast(pl.Utf8)
                                    .alias("object"),
                                ],
                            ),
                        )

                # reference_date
                if "reference_date" in ref_unique.columns:
                    filtered = ref_unique.filter(pl.col("reference_date").is_not_null())
                    if len(filtered) > 0:
                        literal_dfs.append(
                            filtered.select(
                                [
                                    pl.col("_ref_uri").alias("subject"),
                                    pl.lit(f"{wdt_ns}P577").alias("predicate"),
                                    pl.col("reference_date")
                                    .cast(pl.Utf8)
                                    .alias("object"),
                                ],
                            ),
                        )

            # Concatenate all DataFrames
            iri_df = pl.concat(iri_dfs) if iri_dfs else None
            literal_df = pl.concat(literal_dfs) if literal_dfs else None

            return iri_df, literal_df

    # ========================================================================
    # FACADE
    # ========================================================================

    class LOTUSExplorer:
        """Facade coordinating search, filtering, and export workflows."""

        def __init__(self, config: dict, is_wasm: bool = False):
            """Initialize application services.

            Parameters
            ----------
            config : dict
                Runtime configuration dictionary.
            is_wasm : bool
                Whether the app runs in a WASM environment.

            """
            self.config = config
            self.is_wasm = is_wasm
            self.memory = MemoryManager(is_wasm)
            # Use normalized mode in WASM for memory efficiency
            self.query_service = WikidataQueryService(
                config["qlever_endpoint"],
                use_normalized=is_wasm,
            )
            self.transform_service = DataTransformService()
            self.filter_service = FilterService()
            self.taxon_service = TaxonResolutionService(config["qlever_endpoint"])

        def resolve_taxon(
            self,
            taxon_input: str,
        ) -> tuple[str | None, mo.Html | None]:
            """Resolve taxon input through the taxon resolution service."""
            return self.taxon_service.resolve(taxon_input)

        def search(
            self,
            criteria: SearchCriteria,
            qid: str,
        ) -> tuple[pl.LazyFrame, DatasetStats]:
            """Run query, transformation, and filtering for one search request."""
            raw_data = self.query_service.fetch_compounds(
                qid,
                criteria.smiles,
                criteria.smiles_search_type,
                criteria.smiles_threshold,
            )
            transformed_data = self.transform_service.apply_standard_transforms(
                raw_data,
                from_normalized=self.is_wasm,  # Skip redundant transforms in WASM mode
            )
            filtered_data = self.filter_service.apply_filters(
                transformed_data,
                criteria,
            )
            stats = DatasetStats.from_lazyframe(filtered_data)
            return filtered_data, stats

        def export(self, df: pl.LazyFrame, format: str, **kwargs) -> bytes:
            """Export query results using the requested output format."""
            if format == "csv":
                strategy = CSVExportStrategy(self.memory)
            elif format == "json":
                strategy = JSONExportStrategy(self.memory)
            elif format == "ttl":
                strategy = TTLExportStrategy(
                    self.memory,
                    kwargs["taxon_input"],
                    kwargs["qid"],
                    kwargs.get("filters", {}),
                )
            else:
                raise ValueError(f"Unknown format: {format}")
            return strategy.export(df)

        def prepare_export_dataframe(
            self,
            df: pl.LazyFrame,
            include_rdf_ref: bool = False,
        ) -> pl.LazyFrame:
            """Project internal columns to the public export schema."""
            exprs = [
                pl.col("name").alias("compound_name"),
                pl.col("smiles").alias("compound_smiles"),
                pl.col("inchikey").alias("compound_inchikey"),
                pl.col("mass").alias("compound_mass"),
                pl.col("mf").alias("molecular_formula"),
                pl.col("taxon_name"),
                pl.col("ref_title").alias("reference_title"),
                pl.col("ref_doi").alias("reference_doi"),
                pl.col("pub_date").alias("reference_date"),
                pl.concat_str([pl.lit("Q"), pl.col("compound").cast(pl.Utf8)]).alias(
                    "compound_qid",
                ),
                pl.concat_str([pl.lit("Q"), pl.col("taxon").cast(pl.Utf8)]).alias(
                    "taxon_qid",
                ),
                pl.concat_str([pl.lit("Q"), pl.col("reference").cast(pl.Utf8)]).alias(
                    "reference_qid",
                ),
            ]

            if "statement" in df.collect_schema().names():
                exprs.append(
                    pl.col("statement")
                    .cast(pl.Utf8)
                    .str.replace(WIKIDATA_STATEMENT_PREFIX, "", literal=True)
                    .alias("statement_id"),
                )

            if include_rdf_ref and "ref" in df.collect_schema().names():
                exprs.append(pl.col("ref"))

            return df.select(exprs)

        def build_display_dataframe(self, df: pl.LazyFrame, limit: int) -> pl.DataFrame:
            """Build a collected dataframe formatted for UI display."""
            df = df.select(
                [
                    pl.col("smiles").alias("Compound Depiction"),
                    pl.col("name").alias("Compound Name"),
                    pl.col("smiles").alias("Compound SMILES"),
                    pl.col("inchikey").alias("Compound InChIKey"),
                    pl.col("mf").alias("Compound Molecular Formula"),
                    pl.col("mass").alias("Compound Mass"),
                    pl.col("taxon_name").alias("Taxon Name"),
                    pl.col("ref_title").alias("Reference Title"),
                    pl.col("pub_date").alias("Reference Date"),
                    pl.col("ref_doi").alias("Reference DOI"),
                    pl.col("compound").alias("Compound QID"),
                    pl.col("taxon").alias("Taxon QID"),
                    pl.col("reference").alias("Reference QID"),
                    pl.col("statement").alias("Statement"),
                ],
            )
            return df.limit(limit).collect() if limit else df.collect()

        def compute_hashes(
            self,
            qid: str,
            taxon_input: str,
            filters: dict,
            df: pl.LazyFrame,
        ) -> tuple[str, str]:
            """Compute query and result hashes for shareable identifiers."""
            # Normalize components for hash (use consistent representation)
            normalized_qid = qid if qid and qid.strip() else "*"
            normalized_taxon = (
                taxon_input if taxon_input and taxon_input.strip() else ""
            )

            query_components = [normalized_qid, normalized_taxon]
            if filters:
                query_components.append(json.dumps(filters, sort_keys=True))

            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8"),
            ).hexdigest()

            result_hasher = hashlib.sha256()
            try:
                unique_compounds: pl.DataFrame = (
                    df.select(pl.col("compound"))
                    .drop_nulls()
                    .unique()
                    .sort("compound")
                    .collect()
                )
                # Stream in batches instead of full list
                for batch in unique_compounds.iter_slices(1000):
                    for val in batch["compound"]:
                        if val:
                            result_hasher.update(str(val).encode("utf-8"))
                del unique_compounds
                if self.memory.is_wasm:
                    gc.collect()
            except Exception:
                pass

            return query_hash, result_hasher.hexdigest()

        def create_metadata(
            self,
            stats: DatasetStats,
            taxon_input: str,
            qid: str,
            filters: dict,
            query_hash: str,
            result_hash: str,
        ) -> dict:
            """Create rich Schema.org compliant metadata.

            Parameters
            ----------
            stats : DatasetStats
                Stats.
            taxon_input : str
                Taxon input.
            qid : str
                Qid.
            filters : dict
                Filters.
            query_hash : str
                Query hash.
            result_hash : str
                Result hash.

            Returns
            -------
            dict
                Dictionary containing metadata.

            """
            # Normalize taxon information
            effective_taxon = self._normalize_taxon_display(taxon_input, qid)
            taxon_description = self._get_taxon_description(taxon_input, qid, filters)

            # Build dataset name
            smiles_info = filters.get("chemical_structure", {})
            if smiles_info:
                search_type = smiles_info.get("search_type", "substructure")
                dataset_name = (
                    f"LOTUS Data - {search_type.title()} search in {effective_taxon}"
                )
                description = f"Chemical compounds from {taxon_description}"
                description += f". Retrieved via LOTUS Wikidata Explorer with {search_type} chemical search (SACHEM/IDSM)."
            else:
                dataset_name = f"LOTUS Data - {effective_taxon}"
                description = f"Chemical compounds from {taxon_description}"
                description += ". Retrieved via LOTUS Wikidata Explorer."

            metadata: dict[str, Any] = {
                "@context": "https://schema.org/",
                "@type": "Dataset",
                "name": dataset_name,
                "description": description,
                "version": CONFIG["app_version"],
                "dateCreated": datetime.now().isoformat(),
                "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                "creator": {
                    "@type": "SoftwareApplication",
                    "name": CONFIG["app_name"],
                    "version": CONFIG["app_version"],
                    "url": CONFIG["app_url"],
                },
                "provider": [
                    {
                        "@type": "Organization",
                        "name": "LOTUS Initiative",
                        "url": "https://www.wikidata.org/wiki/Q104225190",
                    },
                    {
                        "@type": "Organization",
                        "name": "Wikidata",
                        "url": "http://www.wikidata.org/",
                    },
                ],
                "citation": [
                    {
                        "@type": "ScholarlyArticle",
                        "name": "LOTUS initiative",
                        "identifier": "https://doi.org/10.7554/eLife.70780",
                    },
                ],
                "distribution": [
                    {
                        "@type": "DataDownload",
                        "encodingFormat": "text/csv",
                        "contentUrl": "data:text/csv",
                    },
                    {
                        "@type": "DataDownload",
                        "encodingFormat": "application/json",
                        "contentUrl": "data:application/json",
                    },
                    {
                        "@type": "DataDownload",
                        "encodingFormat": "text/turtle",
                        "contentUrl": "data:text/turtle",
                    },
                ],
                "numberOfRecords": stats.n_entries,
                "variablesMeasured": [
                    "compound_name",
                    "compound_smiles",
                    "compound_inchikey",
                    "compound_mass",
                    "molecular_formula",
                    "taxon_name",
                    "reference_title",
                    "reference_doi",
                    "reference_date",
                    "compound_qid",
                    "taxon_qid",
                    "reference_qid",
                ],
                "search_parameters": {
                    "taxon": effective_taxon,
                    "taxon_qid": qid if qid and qid != "*" else None,
                },
            }
            if smiles_info:
                metadata["provider"].append(
                    {
                        "@type": "Organization",
                        "name": "IDSM",
                        "url": "https://idsm.elixir-czech.cz/",
                    },
                )
                metadata["chemical_search_service"] = {
                    "name": "SACHEM",
                    "provider": "IDSM",
                    "endpoint": "https://idsm.elixir-czech.cz/sparql/endpoint/",
                }
                structure_value = smiles_info.get("smiles") or ""
                structure_is_multiline = (
                    "\n" in structure_value or "\r" in structure_value
                )
                structure_query: dict[str, Any] = {
                    "param_key": "structure",
                    "legacy_param_key": "smiles",
                    "search_type": smiles_info.get("search_type", "substructure"),
                    "input_format": "molfile" if structure_is_multiline else "smiles",
                }
                if "similarity_threshold" in smiles_info:
                    structure_query["similarity_threshold"] = smiles_info[
                        "similarity_threshold"
                    ]
                if structure_is_multiline:
                    structure_query["query_preview"] = structure_value
                    structure_query["query_length"] = len(structure_value)
                else:
                    structure_query["query_text"] = structure_value
                search_params: dict[str, Any] = metadata["search_parameters"]
                search_params["structure_query"] = structure_query
            if filters:
                search_params_f: dict[str, Any] = metadata["search_parameters"]
                search_params_f["filters"] = filters
            metadata["sparql_endpoint"] = {
                "url": self.config["qlever_endpoint"],
                "name": "QLever Wikidata",
                "description": "Fast SPARQL endpoint for Wikidata",
            }
            metadata["provenance"] = {
                "query_hash": {
                    "algorithm": "SHA-256",
                    "value": query_hash,
                },
                "result_hash": {
                    "algorithm": "SHA-256",
                    "value": result_hash,
                },
                "dataset_uri": f"urn:hash:sha256:{result_hash}",
            }

            return metadata

        def _normalize_taxon_display(self, taxon_input: str, qid: str) -> str:
            """Get display-friendly taxon name.

            Parameters
            ----------
            taxon_input : str
                Taxon input.
            qid : str
                Qid.

            Returns
            -------
            str
                String representation of normalize taxon display.

            """
            if not taxon_input or taxon_input.strip() == "":
                return "any taxon"
            elif taxon_input == "*":
                return "all taxa"
            else:
                return taxon_input

        def _get_taxon_description(
            self,
            taxon_input: str,
            qid: str,
            filters: dict,
        ) -> str:
            """Get taxon description for metadata.

            Parameters
            ----------
            taxon_input : str
                Taxon input.
            qid : str
                Qid.
            filters : dict
                Filters.

            Returns
            -------
            str
                String representation of get taxon description.

            """
            smiles_info = filters.get("chemical_structure", {})

            if not taxon_input or taxon_input.strip() == "":
                if smiles_info:
                    return "any taxon (structure-only search)"
                else:
                    return "any taxon"
            elif taxon_input == "*":
                return "all taxa (Biota, Wikidata QID: Q2382443)"
            elif qid and qid != "*":
                return f"taxon {taxon_input} (Wikidata QID: {qid})"
            else:
                return f"taxon {taxon_input}"

        def create_citation(self, taxon_input: str) -> str:
            """Generate citation handling None/empty taxon.

            Parameters
            ----------
            taxon_input : str
                Taxon input.

            Returns
            -------
            str
                String representation of citation.

            """
            current_date = datetime.now().strftime("%B %d, %Y")

            # Normalize taxon for citation
            if not taxon_input or taxon_input.strip() == "":
                taxon_display = "any taxon (structure-only search)"
            elif taxon_input == "*":
                taxon_display = "all taxa"
            else:
                taxon_display = taxon_input

            return f"""
            ## How to Cite This Data

            ### Dataset Citation
            LOTUS Initiative via Wikidata. ({datetime.now().year}). *Data for {taxon_display}*.
            Retrieved from LOTUS Wikidata Explorer on {current_date}.
            License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

            ### LOTUS Initiative Publication
            Rutz A, Sorokina M, Galgonek J, et al. (2022). The LOTUS initiative for open knowledge
            management in natural products research. *eLife* **11**:e70780.
            DOI: [10.7554/eLife.70780](https://doi.org/10.7554/eLife.70780)
            """

        def build_shareable_url(self, criteria: SearchCriteria) -> str:
            """Build shareable URL handling None/empty taxon.

            Parameters
            ----------
            criteria : SearchCriteria
                Criteria.

            Returns
            -------
            str
                String representation of shareable url.

            """
            params = {}

            # Only add taxon if it's not empty (empty means structure-only search)
            if criteria.taxon and criteria.taxon.strip() and criteria.taxon != "":
                params["taxon"] = criteria.taxon

            if criteria.smiles:
                params["structure"] = criteria.smiles
                # Keep legacy key for backward compatibility when single-line.
                if "\n" not in criteria.smiles and "\r" not in criteria.smiles:
                    params["smiles"] = criteria.smiles
                params["smiles_search_type"] = criteria.smiles_search_type
                params["structure_search_type"] = criteria.smiles_search_type
                if criteria.smiles_search_type == "similarity":
                    params["smiles_threshold"] = str(criteria.smiles_threshold)

            if criteria.has_mass_filter():
                params["mass_filter"] = "true"
                params["mass_min"] = str(criteria.mass_range[0])
                params["mass_max"] = str(criteria.mass_range[1])

            if criteria.has_year_filter():
                params["year_filter"] = "true"
                params["year_start"] = str(criteria.year_range[0])
                params["year_end"] = str(criteria.year_range[1])

            if params:
                return f"?{urllib.parse.urlencode(params)}"
            return ""

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def generate_filename(
        taxon_name: str,
        file_type: str,
        filters: dict | None = None,
    ) -> str:
        """Generate safe filename handling None/empty taxon.

        Parameters
        ----------
        taxon_name : str
            Taxon name.
        file_type : str
            File .
        filters : dict | None
            None. Default is None.

        Returns
        -------
        str
            String representation of filename.

        """
        # Normalize taxon for filename
        if not taxon_name or taxon_name.strip() == "":
            safe_name = "any_taxon"
        elif taxon_name == "*":
            safe_name = "all_taxa"
        else:
            safe_name = (
                taxon_name.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
                .replace("*", "star")
                .replace("?", "")
                .replace('"', "")
                .replace("<", "")
                .replace(">", "")
                .replace("|", "")
            )

        date_str = datetime.now().strftime("%Y%m%d")

        # Add filter info to filename if present
        if filters and filters.get("chemical_structure"):
            search_type = filters["chemical_structure"].get(
                "search_type",
                "substructure",
            )
            return f"{date_str}_lotus_{safe_name}_{search_type}.{file_type}"

        return f"{date_str}_lotus_{safe_name}.{file_type}"


@app.cell
def md_title():
    """Render the application title."""
    mo.md("""
    # LOTUS Wikidata Explorer
    """)
    return


@app.cell
def ui_help():
    """Render help text with API usage examples."""
    mo.accordion(
        {
            "Help & API": mo.md("""
    **Search:** Enter a taxon name (e.g., *Gentiana lutea*) and/or a structure in SMILES or Molfile (V2000/V3000).

    **URL API:** `?taxon=Salix&structure=CC(=O)Oc1ccccc1C(=O)O`
    """),
        },
    )
    return


@app.cell
def url_api_defaults():
    """Parse URL parameters and derive initial UI defaults."""
    query_params = mo.query_params()

    def _to_float(value: str | list[str] | None, default: float) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value if isinstance(value, str) else value[0])
        except (TypeError, ValueError):
            return default

    _structure_value = query_params["structure"] or query_params["smiles"] or ""
    search_type = (
        query_params["structure_search_type"]
        or query_params["smiles_search_type"]
        or "substructure"
    )
    if search_type not in {"substructure", "similarity"}:
        search_type = "substructure"

    url_api_defaults = {
        "taxon": query_params["taxon"] or "Gentiana lutea",
        "structure": _structure_value,
        "search_type": search_type,
        "threshold": _to_float(query_params["smiles_threshold"], 0.8),
    }
    return (url_api_defaults,)


@app.cell
def ui_search_inputs(url_api_defaults):
    """Create core search and filter input widgets."""
    taxon_input = mo.ui.text(
        value=url_api_defaults["taxon"],
        label="Taxon Name or QID",
        full_width=True,
    )
    smiles_input = mo.ui.text_area(
        value=url_api_defaults["structure"],
        label="SMILES or Molfile",
        placeholder="Paste a SMILES string or a Molfile (V2000/V3000)",
        full_width=True,
    )
    smiles_search_type = mo.ui.dropdown(
        options=["substructure", "similarity"],
        value=url_api_defaults["search_type"],
    )
    smiles_threshold = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.05,
        value=url_api_defaults["threshold"],
        label="Threshold",
    )
    mass_filter = mo.ui.checkbox(label="Mass filter", value=False)
    mass_min = mo.ui.number(value=0, start=0, stop=10000, label="Min (Da)")
    mass_max = mo.ui.number(value=2000, start=0, stop=10000, label="Max (Da)")
    year_filter = mo.ui.checkbox(label="Year filter", value=False)
    year_start = mo.ui.number(value=1900, start=1700, stop=YEAR, label="From")
    year_end = mo.ui.number(value=YEAR, start=1700, stop=YEAR, label="To")
    formula_filter = mo.ui.checkbox(label="Formula filter", value=False)
    exact_formula = mo.ui.text(value="", label="Exact Formula", placeholder="C15H10O5")
    c_min = mo.ui.number(value=0, start=0, stop=100, label="C min")
    c_max = mo.ui.number(value=100, start=0, stop=100, label="C max")
    h_min = mo.ui.number(value=0, start=0, stop=200, label="H min")
    h_max = mo.ui.number(value=200, start=0, stop=200, label="H max")
    n_min = mo.ui.number(value=0, start=0, stop=50, label="N min")
    n_max = mo.ui.number(value=50, start=0, stop=50, label="N max")
    o_min = mo.ui.number(value=0, start=0, stop=50, label="O min")
    o_max = mo.ui.number(value=50, start=0, stop=50, label="O max")
    p_min = mo.ui.number(value=0, start=0, stop=20, label="P min")
    p_max = mo.ui.number(value=20, start=0, stop=20, label="P max")
    s_min = mo.ui.number(value=0, start=0, stop=20, label="S min")
    s_max = mo.ui.number(value=20, start=0, stop=20, label="S max")
    f_state = mo.ui.dropdown(
        options=["allowed", "required", "excluded"],
        value="allowed",
        label="F",
    )
    cl_state = mo.ui.dropdown(
        options=["allowed", "required", "excluded"],
        value="allowed",
        label="Cl",
    )
    br_state = mo.ui.dropdown(
        options=["allowed", "required", "excluded"],
        value="allowed",
        label="Br",
    )
    i_state = mo.ui.dropdown(
        options=["allowed", "required", "excluded"],
        value="allowed",
        label="I",
    )

    run_button = mo.ui.run_button(label="Search Wikidata")
    return (
        br_state,
        c_max,
        c_min,
        cl_state,
        exact_formula,
        f_state,
        formula_filter,
        h_max,
        h_min,
        i_state,
        mass_filter,
        mass_max,
        mass_min,
        n_max,
        n_min,
        o_max,
        o_min,
        p_max,
        p_min,
        run_button,
        s_max,
        s_min,
        smiles_input,
        smiles_search_type,
        smiles_threshold,
        taxon_input,
        year_end,
        year_filter,
        year_start,
    )


@app.cell
def ketcher_helper():
    """Render optional Ketcher helper content."""
    mo.accordion(
        {
            "✏️ Ketcher Structure Editor: draw or look up a structure": mo.vstack(
                [
                    mo.Html("""
                <iframe
                  src="public/standalone/index.html"
                  style="width:100%; height:800px; border:1px solid #ccc; border-radius:6px;"
                ></iframe>
            """),
                    mo.callout(
                        mo.md(
                            "**Copy SMILES** - *Edit → Copy as Daylight SMILES* (`Ctrl+Shift+S`), then paste into the structure box",
                        ),
                        kind="info",
                    ),
                ],
                gap=1,
            ),
        },
    )
    return


@app.cell
def ui_search_panel(
    br_state,
    c_max,
    c_min,
    cl_state,
    exact_formula,
    f_state,
    formula_filter,
    h_max,
    h_min,
    i_state,
    mass_filter,
    mass_max,
    mass_min,
    n_max,
    n_min,
    o_max,
    o_min,
    p_max,
    p_min,
    run_button,
    s_max,
    s_min,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    """Assemble and display the complete search control panel."""
    structure_fields = [smiles_input, smiles_search_type]
    if smiles_search_type.value == "similarity":
        structure_fields.append(smiles_threshold)

    main_search = mo.hstack(
        [
            mo.vstack([taxon_input, run_button]),
            mo.vstack(structure_fields),
        ],
        gap=2,
        widths="equal",
        wrap=True,
    )

    filter_row = mo.hstack([mass_filter, year_filter, formula_filter], gap=2, wrap=True)
    filters_ui = [filter_row, main_search]

    if mass_filter.value:
        filters_ui.append(mo.hstack([mass_min, mass_max], gap=2, wrap=True))
    if year_filter.value:
        filters_ui.append(mo.hstack([year_start, year_end], gap=2, wrap=True))
    if formula_filter.value:
        filters_ui.extend(
            [
                exact_formula,
                mo.hstack([c_min, c_max, h_min, h_max, n_min, n_max], gap=1, wrap=True),
                mo.hstack([o_min, o_max, p_min, p_max, s_min, s_max], gap=1, wrap=True),
                mo.hstack([f_state, cl_state, br_state, i_state], gap=1, wrap=True),
            ],
        )

    mo.vstack(filters_ui, gap=1)
    return


@app.cell
def execute_search(
    br_state,
    c_max,
    c_min,
    cl_state,
    exact_formula,
    f_state,
    formula_filter,
    h_max,
    h_min,
    i_state,
    mass_filter,
    mass_max,
    mass_min,
    n_max,
    n_min,
    o_max,
    o_min,
    p_max,
    p_min,
    run_button,
    s_max,
    s_min,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    """Execute search workflow and return query outputs."""
    def _backend_error_callout(
        context: str,
        exc: Exception,
        molfile_hint: bool = False,
    ):
        error_text = str(exc).strip()

        status = None
        for pattern in (
            r"HTTP\s+status\s*(\d{3})",
            r"HTTP\s+Error\s*(\d{3})",
            r"status[=: ]+\s*(\d{3})",
        ):
            status_match = re.search(pattern, error_text, re.IGNORECASE)
            if status_match:
                status = status_match.group(1)
                break

        status_text = f" (HTTP {status})" if status else ""

        message_parts = [
            f"**{context} {status_text}.**",
            "QLever returned a server error. Please retry in a moment.",
        ]
        if molfile_hint:
            message_parts.append(
                "If this is a Molfile query, rerunning once may help for transient upstream failures.",
            )
        message_parts.append(
            f"Raw upstream error:\n\n```text\n{error_text[:1200]}\n```",
        )

        return mo.callout(mo.md("\n\n".join(message_parts)), kind="warn")

    if not run_button.value:
        lotus, results, stats, qid, criteria, query_hash, result_hash, taxon_warning = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    else:
        start_time = time.time()
        lotus = LOTUSExplorer(CONFIG, IS_PYODIDE)

        if not taxon_input.value.strip() and not smiles_input.value.strip():
            mo.stop(True, "Need taxon or structure")
        else:
            try:
                qid, taxon_warning = lotus.resolve_taxon(taxon_input.value)
            except (ConnectionError, RuntimeError, TimeoutError) as exc:
                mo.stop(True, _backend_error_callout("Error", exc))

        formula_filt = None
        if formula_filter.value:
            formula_filt = create_filters(
                exact_formula=exact_formula.value,
                c_min=c_min.value,
                c_max=c_max.value,
                h_min=h_min.value,
                h_max=h_max.value,
                n_min=n_min.value,
                n_max=n_max.value,
                o_min=o_min.value,
                o_max=o_max.value,
                p_min=p_min.value,
                p_max=p_max.value,
                s_min=s_min.value,
                s_max=s_max.value,
                f_state=f_state.value,
                cl_state=cl_state.value,
                br_state=br_state.value,
                i_state=i_state.value,
            )

        criteria = SearchCriteria(
            taxon=taxon_input.value,
            smiles=smiles_input.value if smiles_input.value else "",
            smiles_search_type=smiles_search_type.value,
            smiles_threshold=smiles_threshold.value,
            mass_range=(mass_min.value, mass_max.value)
            if mass_filter.value
            else (0.0, 2000.0),
            year_range=(year_start.value, year_end.value)
            if year_filter.value
            else (1900, YEAR),
            formula_filters=formula_filt,
        )

        with mo.status.spinner("Searching..."):
            try:
                results, stats = lotus.search(criteria, qid or "")
            except ValueError as exc:
                mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))
            except (ConnectionError, RuntimeError, TimeoutError) as exc:
                mo.stop(
                    True,
                    _backend_error_callout("Search", exc, molfile_hint=True),
                )

        query_hash, result_hash = lotus.compute_hashes(
            qid or "",
            criteria.taxon,
            criteria.to_filters_dict(),
            results,
        )

        elapsed = round(time.time() - start_time, 2)
        mo.md(f"Query executed in **{elapsed}s**")
    return (
        criteria,
        lotus,
        qid,
        query_hash,
        result_hash,
        results,
        stats,
        taxon_warning,
    )


@app.cell
def display_results(
    criteria,
    lotus,
    qid,
    query_hash,
    result_hash,
    results,
    stats,
    taxon_warning,
):
    """Display result statistics, tables, and preview exports."""
    if results is None or stats is None:
        _out = mo.Html("")
    elif stats.n_entries == 0:
        _out = mo.callout(
            mo.md(f"No compounds found for **{criteria.taxon}**"),
            kind="warn",
        )
    else:
        display_df = lotus.build_display_dataframe(results, CONFIG["table_row_limit"])

        def wrap_image2(smiles: str) -> mo.Html:
            return mo.image(svg_from_smiles(smiles)) if smiles else mo.image("")

        def wrap_qid(qid_val: str | int, color: str | int) -> mo.Html:
            if not qid_val:
                return mo.Html("")
            if qid_val == "*":
                url = "https://qlever.scholia.wiki/taxon/all"
            qid_str = str(qid_val)
            qid_norm = qid_str if qid_str.startswith("Q") else f"Q{qid_str}"
            url = f"https://scholia.toolforge.org/{qid_norm}"
            return mo.Html(
                f'<a href="{url}" style="color:{color};" target="_blank">{qid_norm}</a>',
            )

        def wrap_doi(doi: str) -> mo.Html:
            if not doi:
                return mo.Html("")
            url = f"https://doi.org/{doi}"
            return mo.Html(
                f'<a href="{url}" style="color:{CONFIG["color_hyperlink"]};" target="_blank">{doi}</a>',
            )

        def wrap_statement(statement: str) -> mo.Html:
            if not statement:
                return mo.Html("")
            statement_id = statement.replace(WIKIDATA_STATEMENT_PREFIX, "")
            url = f"https://www.wikidata.org/entity/statement/{statement_id}"
            return mo.Html(
                f'<a href="{url}" style="color:{CONFIG["color_hyperlink"]};" target="_blank">{statement_id}</a>',
            )

        display_taxon = (
            "all taxa"
            if criteria.taxon == "*"
            else criteria.taxon
            if criteria.taxon is not None
            else ""
        )
        if criteria.smiles:
            if "\n" in criteria.smiles or "\r" in criteria.smiles:
                structure_label = "Molfile"
                _structure_value = "multiline input"
            else:
                structure_label = "SMILES"
                _structure_value = criteria.smiles
            display_compound = f"**{structure_label}:** `{_structure_value}` ({criteria.smiles_search_type}) \n\n"
        else:
            display_compound = "\n\n"
        display_hashes = (
            f"**Hashes:** \n\n*Query*: `{query_hash}` - *Results*: `{result_hash}`"
        )
        if display_taxon == "all taxa":
            search_info = mo.md(
                f"**{display_taxon}** ([Scholia](https://scholia.toolforge.org/taxon)) {display_compound} {display_hashes}",
            )
        elif display_taxon == "":
            search_info = mo.md(
                f"{display_compound} {display_hashes}",
            )
        else:
            search_info = mo.md(
                f"**{display_taxon}** ([{qid}](https://scholia.toolforge.org/{qid})) {display_compound} {display_hashes}",
            )

        stats_ui = mo.hstack(
            [
                mo.stat(
                    value=f"{stats.n_compounds:,}",
                    label=pluralize(
                        "Compound",
                        stats.n_compounds,
                        irregular=PLURAL_MAP,
                    ),
                ),
                mo.stat(
                    value=f"{stats.n_taxa:,}",
                    label=pluralize("Taxon", stats.n_taxa, irregular=PLURAL_MAP),
                ),
                mo.stat(
                    value=f"{stats.n_references:,}",
                    label=pluralize(
                        "Reference",
                        stats.n_references,
                        irregular=PLURAL_MAP,
                    ),
                ),
                mo.stat(
                    value=f"{stats.n_entries:,}",
                    label=pluralize("Entry", stats.n_entries, irregular=PLURAL_MAP),
                ),
            ],
            gap=0,
            wrap=True,
        )

        table_ui = mo.ui.table(
            data=display_df,
            format_mapping={
                "Compound Depiction": wrap_image2,
                "Compound QID": lambda x: wrap_qid(x, CONFIG["color_wikidata_red"]),
                "Taxon QID": lambda x: wrap_qid(x, CONFIG["color_wikidata_green"]),
                "Reference QID": lambda x: wrap_qid(x, CONFIG["color_wikidata_blue"]),
                "Reference DOI": wrap_doi,
                "Statement": wrap_statement,
            },
            page_size=CONFIG["page_size_default"],
        )

        export_df_preview = (
            lotus.prepare_export_dataframe(results, include_rdf_ref=False)
            .limit(10)
            .collect()
        )
        metadata = lotus.create_metadata(
            stats,
            criteria.taxon,
            qid,
            criteria.to_filters_dict(),
            query_hash,
            result_hash,
        )
        citation = lotus.create_citation(criteria.taxon)

        tabs_ui = mo.ui.tabs(
            {
                "Display": table_ui,
                "Export View": mo.ui.table(
                    data=export_df_preview,
                    page_size=CONFIG["page_size_export"],
                ),
                "Citation": mo.md(citation),
                "Metadata": mo.md(f"```json\n{json.dumps(metadata, indent=2)}\n```"),
            },
        )

        shareable_url = lotus.build_shareable_url(criteria)

        result_parts = [
            mo.md("## Results"),
            search_info.style(
                style={
                    "overflow-wrap": "anywhere",
                },
            ),
            stats_ui,
        ]
        if shareable_url:
            result_parts.append(
                mo.accordion(
                    {"Share this search": mo.md(f"```\n{shareable_url}\n```")},
                ),
            )
        if taxon_warning:
            result_parts.append(mo.callout(taxon_warning, kind="warn"))
        result_parts.append(tabs_ui)

        _out = mo.vstack(result_parts)

    _out
    return (metadata,)


@app.cell
def generate_downloads(
    criteria,
    lotus,
    metadata,
    qid,
    query_hash,
    result_hash,
    results,
    run_button,
):
    """Create download buttons for CSV, JSON, TTL, and metadata exports."""
    if not run_button.value:
        _out = mo.Html("")
    else:
        export_df = lotus.prepare_export_dataframe(results, include_rdf_ref=False)
        rdf_df = lotus.prepare_export_dataframe(results, include_rdf_ref=True)

        buttons = [
            mo.download(
                label="CSV",
                filename=generate_filename(criteria.taxon, "csv.gz"),
                mimetype="application/gzip",
                data=lambda: compress_if_large(
                    lotus.export(export_df, "csv"),
                    CONFIG["download_embed_threshold_bytes"],
                )[0],
            ),
            mo.download(
                label="JSON",
                filename=generate_filename(criteria.taxon, "json.gz"),
                mimetype="application/gzip",
                data=lambda: compress_if_large(
                    lotus.export(export_df, "json"),
                    CONFIG["download_embed_threshold_bytes"],
                )[0],
            ),
            mo.download(
                label="TTL",
                filename=generate_filename(criteria.taxon, "ttl.gz"),
                mimetype="application/gzip",
                data=lambda: compress_if_large(
                    lotus.export(
                        rdf_df,
                        "ttl",
                        taxon_input=criteria.taxon,
                        qid=qid,
                        filters=criteria.to_filters_dict(),
                    ),
                    CONFIG["download_embed_threshold_bytes"],
                )[0],
            ),
            mo.download(
                label="Metadata",
                filename=f"{query_hash}_{result_hash}_metadata.json",
                mimetype="application/json",
                data=lambda: json.dumps(metadata, indent=2).encode("utf-8"),
            ),
        ]

        _out = mo.vstack(
            [
                mo.md("### Download Data"),
                mo.hstack(buttons, gap=2, wrap=True),
            ],
        )
    _out
    return


@app.cell
def ui_disclaimer():
    """Render runtime-specific disclaimer notices."""
    if IS_PYODIDE:
        _out = mo.callout(
            mo.md(f"""
            **Browser Version Limitations:**
            - Showing only {CONFIG["table_row_limit"]} results due to memory constraints
            - Large exports may fail
            - For unlimited results, run locally:
            ```bash
            uvx marimo run https://adafede.github.io/marimo/apps/lotus_wikidata_explorer.py
            ```
            """),
            kind="warn",
        ).style(
            style={
                "overflow-wrap": "anywhere",
            },
        )
    else:
        _out = mo.Html("")
    _out
    return


@app.cell
def footer():
    """Render footer links for data sources, tools, and licenses."""
    mo.md("""
    ---
    **Data:**
    <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> &
    <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a> |
    **Code:**
    <a href="https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py" style="color:#339966;">lotus_wikidata_explorer.py</a> |
    **External tools:**
    <a href="https://github.com/cdk/depict" style="color:#006699;">CDK Depict</a> &
    <a href="https://idsm.elixir-czech.cz/" style="color:#006699;">IDSM</a> &
    <a href="https://doi.org/10.1186/s13321-018-0282-y" style="color:#006699;">Sachem</a> &
    <a href="https://qlever.dev/wikidata" style="color:#006699;">QLever</a> |
    **License:**
    <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#484848;">CC0 1.0</a> for data &
    <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#484848;">AGPL-3.0</a> for code
    """)
    return


def main():
    """Entry point for CLI and GUI modes."""
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser(description="Export LOTUS data via CLI")
        parser.add_argument("export")
        parser.add_argument("--taxon", help="Taxon name or QID")
        parser.add_argument("--output", "-o", help="Output file")
        parser.add_argument(
            "--format",
            "-f",
            choices=["csv", "json", "ttl"],
            default="csv",
        )
        parser.add_argument(
            "--smiles",
            help="Structure query text (SMILES or Molfile V2000/V3000)",
        )
        parser.add_argument(
            "--smiles-search-type",
            choices=["substructure", "similarity"],
            default="substructure",
        )
        parser.add_argument("--smiles-threshold", type=float, default=0.8)
        parser.add_argument("--year-start", type=int)
        parser.add_argument("--year-end", type=int)
        parser.add_argument("--mass-min", type=float)
        parser.add_argument("--mass-max", type=float)
        parser.add_argument("--formula", help="Exact molecular formula")
        parser.add_argument("--c-min", type=int)
        parser.add_argument("--c-max", type=int)
        parser.add_argument("--h-min", type=int)
        parser.add_argument("--h-max", type=int)
        parser.add_argument("--n-min", type=int)
        parser.add_argument("--n-max", type=int)
        parser.add_argument("--o-min", type=int)
        parser.add_argument("--o-max", type=int)
        parser.add_argument("--p-min", type=int)
        parser.add_argument("--p-max", type=int)
        parser.add_argument("--s-min", type=int)
        parser.add_argument("--s-max", type=int)
        parser.add_argument("--compress", action="store_true")
        parser.add_argument("--show-metadata", action="store_true")
        parser.add_argument("--export-metadata", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        args = parser.parse_args()

        try:
            lotus = LOTUSExplorer(CONFIG, False)

            if args.taxon is None or args.taxon.strip() == "":
                if args.smiles:
                    # Structure-only search
                    args.taxon = ""
                    qid = "*"
                else:
                    print("[x] Need taxon or structure", file=sys.stderr)
                    sys.exit(1)
            elif args.taxon == "*":
                qid = "*"
            else:
                qid, _ = lotus.resolve_taxon(args.taxon)
            if not qid:
                print(f"[x] Taxon not found: {args.taxon}", file=sys.stderr)
                sys.exit(1)

            if args.verbose:
                print(f"Querying: {args.taxon} (QID: {qid})", file=sys.stderr)

            formula_filt = None
            if any(
                [
                    args.formula,
                    args.c_min,
                    args.c_max,
                    args.h_min,
                    args.h_max,
                    args.n_min,
                    args.n_max,
                    args.o_min,
                    args.o_max,
                    args.p_min,
                    args.p_max,
                    args.s_min,
                    args.s_max,
                ],
            ):
                formula_filt = create_filters(
                    exact_formula=args.formula or "",
                    c_min=args.c_min or 0,
                    c_max=args.c_max or None,
                    h_min=args.h_min or 0,
                    h_max=args.h_max or None,
                    n_min=args.n_min or 0,
                    n_max=args.n_max or None,
                    o_min=args.o_min or 0,
                    o_max=args.o_max or None,
                    p_min=args.p_min or 0,
                    p_max=args.p_max or None,
                    s_min=args.s_min or 0,
                    s_max=args.s_max or None,
                    f_state="allowed",
                    cl_state="allowed",
                    br_state="allowed",
                    i_state="allowed",
                )

            criteria = SearchCriteria(
                taxon=args.taxon,
                smiles=args.smiles if args.smiles is not None else "",
                smiles_search_type=args.smiles_search_type,
                smiles_threshold=args.smiles_threshold,
                mass_range=(args.mass_min or 0.0, args.mass_max or 2000.0)
                if args.mass_min or args.mass_max
                else (0.0, 2000.0),
                year_range=(args.year_start or 1900, args.year_end or YEAR)
                if args.year_start or args.year_end
                else (1900, YEAR),
                formula_filters=formula_filt,
            )

            results, stats = lotus.search(criteria, qid)

            if stats.n_entries == 0:
                print("[x] No data found", file=sys.stderr)
                sys.exit(1)

            if args.verbose:
                print(f"Found {stats.n_entries:,} entries", file=sys.stderr)

            query_hash, result_hash = lotus.compute_hashes(
                qid,
                args.taxon,
                criteria.to_filters_dict(),
                results,
            )

            if args.show_metadata:
                metadata = lotus.create_metadata(
                    stats,
                    args.taxon,
                    qid,
                    criteria.to_filters_dict(),
                    query_hash,
                    result_hash,
                )
                print(json.dumps(metadata, indent=2))

            if args.format in ["csv", "json"]:
                export_df = lotus.prepare_export_dataframe(
                    results,
                    include_rdf_ref=False,
                )
            else:
                export_df = lotus.prepare_export_dataframe(
                    results,
                    include_rdf_ref=True,
                )

            data = lotus.export(
                export_df,
                args.format,
                taxon_input=args.taxon,
                qid=qid,
                filters=criteria.to_filters_dict(),
            )

            if args.compress:
                import gzip

                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                    gz.write(data)
                data = buffer.getvalue()

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(data)
                if args.verbose:
                    print(f"[+] Exported to: {output_path}", file=sys.stderr)

                if args.export_metadata:
                    metadata = lotus.create_metadata(
                        stats,
                        args.taxon,
                        qid,
                        criteria.to_filters_dict(),
                        query_hash,
                        result_hash,
                    )
                    metadata_path = output_path.with_suffix(
                        output_path.suffix + ".metadata.json",
                    )
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    if args.verbose:
                        print(f"[+] Metadata: {metadata_path}", file=sys.stderr)
            else:
                sys.stdout.buffer.write(data)

        except Exception as e:
            print(f"[x] Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)
    else:
        app.run()


if __name__ == "__main__":
    main()
