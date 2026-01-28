# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "great-tables==0.20.0",
#     "marimo",
#     "polars==1.37.1",
#     "rdflib==7.5.0",
# ]
# [tool.marimo.runtime]
# output_max_bytes = 300_000_000
# ///

"""
LOTUS Wikidata Explorer - Refactored

Clean, SOLID, DRY architecture with proper separation of concerns.

Copyright (C) 2026 Adriano Rutz
License: AGPL-3.0
"""

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")


# ============================================================================
# SETUP: Core Domain & Services
# ============================================================================

with app.setup:
    import marimo as mo
    import polars as pl
    import io
    import json
    import re
    import time
    import hashlib
    import sys
    import urllib.parse
    import gc
    from dataclasses import dataclass, field, replace
    from datetime import datetime
    from rdflib import Graph, Literal, URIRef, BNode
    from rdflib.namespace import RDF, RDFS, XSD, DCTERMS
    from typing import Any, Optional, Protocol
    from contextlib import contextmanager
    from abc import ABC, abstractmethod

    # Module imports (same as before)
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
    from modules.knowledge.wikidata.sparql.query_taxon_details import query_taxon_details
    from modules.knowledge.wikidata.sparql.query_sachem import query_sachem
    from modules.knowledge.wikidata.sparql.query_compounds import (
        query_compounds_by_taxon,
        query_all_compounds,
    )
    from modules.net.sparql.execute_with_retry import execute_with_retry
    from modules.net.sparql.parse_response import parse_sparql_response
    from modules.net.sparql.values_clause import values_clause
    from modules.chem.cdk.depict.svg_from_smiles import svg_from_smiles
    from modules.knowledge.rdf.graph.add_literal import add_literal
    from modules.knowledge.rdf.namespace.wikidata import WIKIDATA_NAMESPACES
    from modules.text.formula.element_config import ELEMENT_DEFAULTS
    from modules.ui.marimo.wrap_html import wrap_html
    from modules.ui.marimo.wrap_image import wrap_image
    from modules.io.compress.if_large import compress_if_large

    IS_PYODIDE = "pyodide" in sys.modules
    if IS_PYODIDE:
        import pyodide_http
        pyodide_http.patch_all()

    # Configuration
    CONFIG = {
        "app_version": "0.2.0",
        "app_name": "LOTUS Wikidata Explorer",
        "qlever_endpoint": "https://qlever.dev/api/wikidata",
        "wikidata_endpoint": "https://query.wikidata.org/sparql",
        "idsm_endpoint": "https://idsm.elixir-czech.cz/sparql/endpoint/",
        "table_row_limit": 1_000,
        "download_embed_threshold_bytes": 500_000,
        "color_hyperlink": "#3377c4",
        "color_wikidata_blue": "#006699",
        "color_wikidata_green": "#339966",
        "color_wikidata_red": "#990000",
        "page_size_default": 10,
        "year_default_start": 1_900,
        "mass_default_min": 0,
        "mass_default_max": 2_000,
        "mass_ui_max": 10_000,
    }

    PLURAL_MAP = {"Entry": "Entries", "Taxon": "Taxa"}

    # ========================================================================
    # DOMAIN MODELS (Immutable, Pure Data)
    # ========================================================================

    @dataclass(frozen=True)
    class SearchCriteria:
        """Immutable search parameters."""
        taxon: str = ""
        smiles: str = ""
        smiles_search_type: str = "substructure"
        smiles_threshold: float = 0.8
        mass_range: tuple[float, float] = (0.0, 2000.0)
        year_range: tuple[int, int] = (1900, datetime.now().year)
        formula_filters: Optional[FormulaFilters] = None

        def has_mass_filter(self) -> bool:
            return self.mass_range != (0.0, 2000.0)

        def has_year_filter(self) -> bool:
            return self.year_range != (1900, datetime.now().year)

        def to_filters_dict(self) -> dict:
            """Convert to filters dictionary for metadata."""
            filters = {}
            if self.smiles:
                filters["chemical_structure"] = {
                    "smiles": self.smiles,
                    "search_type": self.smiles_search_type,
                }
                if self.smiles_search_type == "similarity":
                    filters["chemical_structure"]["similarity_threshold"] = self.smiles_threshold
            if self.has_mass_filter():
                filters["mass"] = {"min": self.mass_range[0], "max": self.mass_range[1]}
            if self.has_year_filter():
                filters["publication_year"] = {"start": self.year_range[0], "end": self.year_range[1]}
            if self.formula_filters:
                formula_dict = serialize_filters(self.formula_filters)
                if formula_dict:
                    filters["molecular_formula"] = formula_dict
            return filters

    @dataclass(frozen=True)
    class DatasetStats:
        """Dataset statistics."""
        n_compounds: int
        n_taxa: int
        n_references: int
        n_entries: int

        @classmethod
        def from_lazyframe(cls, df: pl.LazyFrame) -> "DatasetStats":
            """Compute stats without materializing data."""
            stats = df.select([
                pl.col("compound").n_unique().alias("n_compounds"),
                pl.col("taxon").n_unique().alias("n_taxa"),
                pl.col("reference").n_unique().alias("n_refs"),
                pl.len().alias("n_entries"),
            ]).collect()

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
        """Centralized memory management with context managers."""

        def __init__(self, is_wasm: bool):
            self.is_wasm = is_wasm

        @contextmanager
        def scope(self, *objects):
            """Context manager for automatic cleanup."""
            try:
                yield
            finally:
                for obj in objects:
                    if obj is not None:
                        del obj
                if self.is_wasm:
                    gc.collect()

        def get_batch_size(self, format: str) -> int:
            """Get optimal batch size for export format."""
            sizes = {
                "csv": (2000, 10000),
                "json": (1000, 5000),
                "rdf": (500, 2000),
            }
            wasm_size, desktop_size = sizes.get(format, (1000, 5000))
            return wasm_size if self.is_wasm else desktop_size

    # ========================================================================
    # DATA SERVICES (Single Responsibility)
    # ========================================================================

    class WikidataQueryService:
        """Responsible for executing Wikidata queries."""

        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def fetch_compounds(
                self,
                qid: str,
                smiles: Optional[str] = None,
                smiles_search_type: str = "substructure",
                smiles_threshold: float = 0.8,
        ) -> pl.LazyFrame:
            """Execute query and return lazy results."""
            query = self._build_query(qid, smiles, smiles_search_type, smiles_threshold)
            csv_bytes = execute_with_retry(query, self.endpoint)

            if not csv_bytes or csv_bytes.strip() == b"":
                return pl.LazyFrame()

            return pl.scan_csv(io.BytesIO(csv_bytes), low_memory=True, rechunk=False)

        def _build_query(
                self,
                qid: str,
                smiles: Optional[str],
                search_type: str,
                threshold: float,
        ) -> str:
            """Build SPARQL query based on parameters."""
            if smiles:
                return query_sachem(
                    escaped_smiles=validate_and_escape(smiles),
                    search_type=search_type,
                    threshold=threshold,
                    taxon_qid=qid if qid != "*" else None,
                )
            elif qid == "*":
                return query_all_compounds()
            else:
                return query_compounds_by_taxon(qid)

    class DataTransformService:
        """Responsible for data transformations."""

        @staticmethod
        def apply_standard_transforms(df: pl.LazyFrame) -> pl.LazyFrame:
            """Apply all standard transformations."""
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
            return df.rename({
                "compoundLabel": "name",
                "compound_inchikey": "inchikey",
                "ref_qid": "reference",
                "ref_date": "pub_date",
                "compound_mass": "mass",
                "compound_formula": "mf",
            })

        @staticmethod
        def combine_smiles(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns([
                pl.coalesce(["compound_smiles_iso", "compound_smiles_conn"]).alias("smiles")
            ])

        @staticmethod
        def extract_doi(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns([
                pl.when(pl.col("ref_doi").str.starts_with("http"))
                .then(pl.col("ref_doi").str.split("doi.org/").list.last())
                .otherwise(pl.col("ref_doi"))
                .alias("ref_doi")
            ])

        @staticmethod
        def parse_dates(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns([
                pl.when(pl.col("pub_date").is_not_null() & (pl.col("pub_date") != ""))
                .then(
                    pl.col("pub_date")
                    .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
                    .dt.date()
                )
                .otherwise(None)
                .alias("pub_date")
            ])

        @staticmethod
        def cast_types(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns([pl.col("mass").cast(pl.Float32, strict=False)])

        @staticmethod
        def drop_old_columns(df: pl.LazyFrame) -> pl.LazyFrame:
            to_drop = ["compound_smiles_iso", "compound_smiles_conn"]
            return df.drop([col for col in to_drop if col in df.collect_schema().names()])

        @staticmethod
        def add_missing_columns(df: pl.LazyFrame) -> pl.LazyFrame:
            required = [
                "compound", "name", "inchikey", "smiles", "taxon_name", "taxon",
                "ref_title", "ref_doi", "reference", "pub_date", "mass", "mf",
                "statement", "ref",
            ]
            missing = [col for col in required if col not in df.collect_schema().names()]
            if missing:
                df = df.with_columns([pl.lit(None).alias(col) for col in missing])
            return df

        @staticmethod
        def deduplicate(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.unique(subset=["compound", "taxon", "reference"], keep="first").sort("name")

    class FilterService:
        """Responsible for filtering data."""

        @staticmethod
        def apply_filters(df: pl.LazyFrame, criteria: SearchCriteria) -> pl.LazyFrame:
            """Apply all filters from criteria."""
            if criteria.has_year_filter():
                df = FilterService.filter_by_year(df, criteria.year_range)
            if criteria.has_mass_filter():
                df = FilterService.filter_by_mass(df, criteria.mass_range)
            if criteria.formula_filters and criteria.formula_filters.is_active():
                df = FilterService.filter_by_formula(df, criteria.formula_filters)
            return df

        @staticmethod
        def filter_by_year(df: pl.LazyFrame, year_range: tuple[int, int]) -> pl.LazyFrame:
            year_start, year_end = year_range
            if year_start:
                df = df.filter(pl.col("pub_date").dt.year() >= year_start)
            if year_end:
                df = df.filter(pl.col("pub_date").dt.year() <= year_end)
            return df

        @staticmethod
        def filter_by_mass(df: pl.LazyFrame, mass_range: tuple[float, float]) -> pl.LazyFrame:
            mass_min, mass_max = mass_range
            if mass_min:
                df = df.filter(pl.col("mass") >= mass_min)
            if mass_max:
                df = df.filter(pl.col("mass") <= mass_max)
            return df

        @staticmethod
        def filter_by_formula(df: pl.LazyFrame, formula_filters: FormulaFilters) -> pl.LazyFrame:
            return df.filter(
                pl.col("mf").map_batches(
                    lambda s: s.map_elements(
                        lambda f: match_filters(f or "", formula_filters),
                        return_dtype=pl.Boolean,
                    )
                )
            )

    # ========================================================================
    # EXPORT STRATEGIES (Strategy Pattern)
    # ========================================================================

    class ExportStrategy(ABC):
        """Base export strategy."""

        def __init__(self, memory: MemoryManager):
            self.memory = memory

        def export(self, df: pl.LazyFrame) -> bytes:
            """Template method for export."""
            return self._to_bytes(df)

        @abstractmethod
        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            pass

    class CSVExportStrategy(ExportStrategy):
        """CSV export."""

        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            df_collected = df.collect()
            buffer = io.BytesIO()
            df_collected.write_csv(buffer)
            result = buffer.getvalue()
            del df_collected, buffer
            if self.memory.is_wasm:
                gc.collect()
            return result

    class JSONExportStrategy(ExportStrategy):
        """JSON export with batching."""

        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            total_rows = df.select(pl.count()).collect()["count"][0]
            batch_size = self.memory.get_batch_size("json")

            buffer = io.BytesIO()
            buffer.write(b'[\n')

            first = True
            for offset in range(0, total_rows, batch_size):
                batch = df.slice(offset, batch_size).collect()

                for record in batch.to_dicts():
                    if not first:
                        buffer.write(b',\n')
                    first = False
                    buffer.write(json.dumps(record, default=str).encode('utf-8'))

                del batch
                if self.memory.is_wasm:
                    gc.collect()

            buffer.write(b'\n]')
            result = buffer.getvalue()
            del buffer
            return result

    class RDFExportStrategy(ExportStrategy):
        """RDF export with batching."""

        def __init__(self, memory: MemoryManager, taxon_input: str, qid: str, filters: dict):
            super().__init__(memory)
            self.taxon_input = taxon_input
            self.qid = qid
            self.filters = filters

        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            df_collected = df.collect()

            # Initialize graph
            g = Graph()
            for prefix, ns in WIKIDATA_NAMESPACES.items():
                g.bind(prefix.lower(), ns)
            g.bind("rdfs", RDFS)
            g.bind("xsd", XSD)
            g.bind("dcterms", DCTERMS)

            # Create dataset URI
            dataset_uri = self._create_dataset_uri(df_collected)

            # Add metadata
            self._add_metadata(g, dataset_uri, len(df_collected))

            # Add compound data in batches
            batch_size = self.memory.get_batch_size("rdf")
            processed_taxa = set()
            processed_refs = set()

            for start_idx in range(0, len(df_collected), batch_size):
                batch = df_collected[start_idx:start_idx + batch_size]
                for row in batch.iter_rows(named=True):
                    self._add_compound_triples(g, row, dataset_uri, processed_taxa, processed_refs)
                del batch

            result = g.serialize(format="turtle").encode('utf-8')
            del df_collected, g
            return result

        def _create_dataset_uri(self, df: pl.DataFrame) -> URIRef:
            """Create content-addressable dataset URI."""
            result_hasher = hashlib.sha256()
            for val in df.select(pl.col("compound_qid").cast(pl.Utf8)).to_series().drop_nulls().unique().sort():
                result_hasher.update(str(val).encode('utf-8'))
            return URIRef(f"urn:hash:sha256:{result_hasher.hexdigest()}")

        def _add_metadata(self, g: Graph, dataset_uri: URIRef, record_count: int):
            """Add dataset metadata."""
            SCHEMA = WIKIDATA_NAMESPACES["SCHEMA"]
            g.add((dataset_uri, RDF.type, SCHEMA.Dataset))
            g.add((dataset_uri, SCHEMA.name, Literal(f"LOTUS Data - {self.taxon_input}")))
            g.add((dataset_uri, SCHEMA.numberOfRecords, Literal(record_count, datatype=XSD.integer)))

        def _add_compound_triples(self, g: Graph, row: dict, dataset_uri: URIRef, processed_taxa: set, processed_refs: set):
            """Add compound triples (simplified for brevity)."""
            WD = WIKIDATA_NAMESPACES["WD"]
            WDT = WIKIDATA_NAMESPACES["WDT"]
            SCHEMA = WIKIDATA_NAMESPACES["SCHEMA"]

            compound_qid = row.get("compound")
            if not compound_qid:
                return

            compound_uri = WD[f"Q{compound_qid}"]
            g.add((dataset_uri, SCHEMA.hasPart, compound_uri))

            if row.get("inchikey"):
                add_literal(g, compound_uri, WDT.P235, row["inchikey"])
            if row.get("smiles"):
                add_literal(g, compound_uri, WDT.P233, row["smiles"])
            if row.get("name"):
                add_literal(g, compound_uri, RDFS.label, row["name"])

    # ========================================================================
    # TAXON RESOLUTION SERVICE
    # ========================================================================

    class TaxonResolutionService:
        """Resolve taxon names to QIDs."""

        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def resolve(self, taxon_input: str) -> tuple[Optional[str], Optional[mo.Html]]:
            """Resolve taxon name to QID."""
            taxon_input = str(taxon_input).strip()

            if not taxon_input:
                return None, None
            if taxon_input == "*":
                return "*", None
            if taxon_input.upper().startswith("Q") and taxon_input[1:].isdigit():
                return taxon_input.upper(), None

            # Search for taxon
            try:
                query = query_taxon_search(taxon_input)
                csv_bytes = execute_with_retry(query, self.endpoint, fallback_endpoint=None)

                if not csv_bytes or not csv_bytes.strip():
                    return None, None

                df = parse_sparql_response(csv_bytes).collect()
                matches = [
                    (extract_from_url(row["taxon"], WIKIDATA_ENTITY_PREFIX), row["taxon_name"])
                    for row in df.iter_rows(named=True)
                    if row.get("taxon") and row.get("taxon_name")
                ]

                if not matches:
                    return None, None

                # Find exact match
                taxon_lower = taxon_input.lower()
                exact_matches = [(qid, name) for qid, name in matches if name.lower() == taxon_lower]

                if len(exact_matches) == 1:
                    return exact_matches[0][0], None

                # Multiple matches - need disambiguation (simplified here)
                return matches[0][0], None

            except Exception:
                return None, None

    # ========================================================================
    # FACADE (Simplified High-Level API)
    # ========================================================================

    class LOTUSExplorer:
        """Simplified facade for LOTUS operations."""

        def __init__(self, config: dict, is_wasm: bool = False):
            self.config = config
            self.memory = MemoryManager(is_wasm)
            self.query_service = WikidataQueryService(config["qlever_endpoint"])
            self.transform_service = DataTransformService()
            self.filter_service = FilterService()
            self.taxon_service = TaxonResolutionService(config["qlever_endpoint"])

        def resolve_taxon(self, taxon_input: str) -> tuple[Optional[str], Optional[mo.Html]]:
            """Resolve taxon name to QID."""
            return self.taxon_service.resolve(taxon_input)

        def search(self, criteria: SearchCriteria, qid: str) -> tuple[pl.LazyFrame, DatasetStats]:
            """Execute search and return results + stats."""
            # Query
            raw_data = self.query_service.fetch_compounds(
                qid,
                criteria.smiles,
                criteria.smiles_search_type,
                criteria.smiles_threshold,
            )

            # Transform
            transformed_data = self.transform_service.apply_standard_transforms(raw_data)

            # Filter
            filtered_data = self.filter_service.apply_filters(transformed_data, criteria)

            # Stats
            stats = DatasetStats.from_lazyframe(filtered_data)

            return filtered_data, stats

        def export(self, df: pl.LazyFrame, format: str, **kwargs) -> bytes:
            """Export data in specified format."""
            if format == "csv":
                strategy = CSVExportStrategy(self.memory)
            elif format == "json":
                strategy = JSONExportStrategy(self.memory)
            elif format == "rdf":
                strategy = RDFExportStrategy(
                    self.memory,
                    kwargs["taxon_input"],
                    kwargs["qid"],
                    kwargs.get("filters", {}),
                )
            else:
                raise ValueError(f"Unknown format: {format}")

            return strategy.export(df)

        def prepare_export_dataframe(self, df: pl.LazyFrame, include_rdf_ref: bool = False) -> pl.LazyFrame:
            """Prepare dataframe for export."""
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
                pl.concat_str([pl.lit("Q"), pl.col("compound").cast(pl.Utf8)]).alias("compound_qid"),
                pl.concat_str([pl.lit("Q"), pl.col("taxon").cast(pl.Utf8)]).alias("taxon_qid"),
                pl.concat_str([pl.lit("Q"), pl.col("reference").cast(pl.Utf8)]).alias("reference_qid"),
            ]

            if "statement" in df.collect_schema().names():
                exprs.append(
                    pl.col("statement")
                    .str.replace(WIKIDATA_STATEMENT_PREFIX, "", literal=True)
                    .alias("statement_id")
                )

            if include_rdf_ref and "ref" in df.collect_schema().names():
                exprs.append(pl.col("ref"))

            return df.select(exprs)

        def build_display_dataframe(self, df: pl.LazyFrame, limit: int) -> pl.DataFrame:
            """Build display DataFrame for UI."""
            df = df.limit(limit).collect() if limit else df.collect()

            return df.select([
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
            ])

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def generate_filename(taxon_name: str, file_type: str, filters: dict = None) -> str:
        """Generate filename for exports."""
        safe_name = taxon_name.replace(" ", "_").replace("/", "_") if taxon_name != "*" else "all_taxa"
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{date_str}_lotus_{safe_name}.{file_type}"

    def create_download_button(data: bytes, filename: str, label: str, mimetype: str):
        """Create download button with compression."""
        compressed, was_compressed = compress_if_large(data, CONFIG["download_embed_threshold_bytes"])
        final_filename = filename + ".gz" if was_compressed else filename
        final_mimetype = "application/gzip" if was_compressed else mimetype
        display_label = label + (" (gzipped)" if was_compressed else "")

        return mo.download(
            data=compressed,
            filename=final_filename,
            label=display_label,
            mimetype=final_mimetype,
        )


# ============================================================================
# UI CELLS
# ============================================================================

@app.cell
def md_title():
    mo.md("# LOTUS Wikidata Explorer")
    return


@app.cell
def ui_help():
    mo.accordion({
        "Help & API": mo.md("""
        **Search:** Enter a taxon name (e.g., *Gentiana lutea*) and/or a SMILES structure, then click Search.
        
        **URL API:** `?taxon=Salix&smiles=CC(=O)Oc1ccccc1C(=O)O`
        """)
    })
    return


@app.cell
def ui_search_inputs():
    """Create search input UI elements."""
    taxon_input = mo.ui.text(
        value="Gentiana lutea",
        label="Taxon Name or QID",
        placeholder="e.g., Gentiana lutea, Q157115, or *",
        full_width=True,
    )

    smiles_input = mo.ui.text(
        value="",
        label="Chemical Structure (SMILES)",
        placeholder="e.g., c1ccccc1",
        full_width=True,
    )

    smiles_search_type = mo.ui.dropdown(
        options=["substructure", "similarity"],
        value="substructure",
        label="Search Type",
    )

    smiles_threshold = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.8,
        label="Similarity Threshold",
    )

    mass_filter = mo.ui.checkbox(label="Mass filter", value=False)
    mass_min = mo.ui.number(value=0, start=0, stop=10000, label="Min (Da)")
    mass_max = mo.ui.number(value=2000, start=0, stop=10000, label="Max (Da)")

    year_filter = mo.ui.checkbox(label="Year filter", value=False)
    year_start = mo.ui.number(value=1900, start=1700, stop=2024, label="From")
    year_end = mo.ui.number(value=2024, start=1700, stop=2024, label="To")

    run_button = mo.ui.run_button(label="Search Wikidata")
    return (
        taxon_input, smiles_input, smiles_search_type, smiles_threshold,
        mass_filter, mass_min, mass_max,
        year_filter, year_start, year_end,
        run_button
    )

@app.cell
def ui_search_panel():
    # Display UI
    structure_fields = [smiles_input, smiles_search_type]
    if smiles_search_type.value == "similarity":
        structure_fields.append(smiles_threshold)

    main_search = mo.hstack([
        mo.vstack([run_button, taxon_input]),
        mo.vstack(structure_fields)
    ], gap=2, widths="equal")

    filter_row = mo.hstack([mass_filter, year_filter], gap=2)
    filters_ui = [main_search, filter_row]

    if mass_filter.value:
        filters_ui.append(mo.hstack([mass_min, mass_max], gap=2))
    if year_filter.value:
        filters_ui.append(mo.hstack([year_start, year_end], gap=2))

    mo.vstack(filters_ui, gap=1)
    return


@app.cell
def execute_search(
        taxon_input, smiles_input, smiles_search_type, smiles_threshold,
        mass_filter, mass_min, mass_max,
        year_filter, year_start, year_end,
        run_button,
):
    """Execute search and return results."""
    if not run_button.value:
        None, None, None, None

    start_time = time.time()

    # Initialize
    lotus = LOTUSExplorer(CONFIG, IS_PYODIDE)

    # Resolve taxon
    qid, taxon_warning = lotus.resolve_taxon(taxon_input.value)
    if not qid:
        mo.stop(True, mo.callout(
            mo.md(f"**Taxon not found:** {taxon_input.value}"),
            kind="warn"
        ))

    # Build criteria
    criteria = SearchCriteria(
        taxon=taxon_input.value,
        smiles=smiles_input.value.strip() if smiles_input.value else "",
        smiles_search_type=smiles_search_type.value,
        smiles_threshold=smiles_threshold.value,
        mass_range=(mass_min.value, mass_max.value) if mass_filter.value else (0.0, 2000.0),
        year_range=(year_start.value, year_end.value) if year_filter.value else (1900, 2024),
    )

    # Execute search
    with mo.status.spinner("Searching..."):
        results, stats = lotus.search(criteria, qid)

    elapsed = round(time.time() - start_time, 2)
    mo.md(f"Query executed in **{elapsed}s**")

    return lotus, results, stats, qid


@app.cell
def display_results(lotus, results, stats, qid, taxon_input):
    """Display search results."""
    if results is None or stats is None:
        mo.Html("")

    if stats.n_entries == 0:
        mo.callout(
            mo.md(f"No compounds found for **{taxon_input.value}**"),
            kind="warn"
        )

    # Build display DataFrame
    display_df = lotus.build_display_dataframe(results, CONFIG["table_row_limit"])

    # Format helpers
    def wrap_image2(smiles: str):
        if not smiles:
            return ""
        return mo.image(svg_from_smiles(smiles))

    def wrap_qid(qid_val: str, color: str):
        if not qid_val:
            return ""
        url = f"https://scholia.toolforge.org/Q{qid_val}"
        return mo.Html(f'<a href="{url}" style="color:{color};" target="_blank">Q{qid_val}</a>')

    # Stats cards
    stats_ui = mo.hstack([
        mo.stat(value=f"{stats.n_compounds:,}", label="Compounds"),
        mo.stat(value=f"{stats.n_taxa:,}", label="Taxa"),
        mo.stat(value=f"{stats.n_references:,}", label="References"),
        mo.stat(value=f"{stats.n_entries:,}", label="Entries"),
    ], gap=0)

    # Table
    table_ui = mo.ui.table(
        data=display_df,
        format_mapping={
            "Compound Depiction": wrap_image2,
            "Compound QID": lambda x: wrap_qid(x, CONFIG["color_wikidata_red"]),
            "Taxon QID": lambda x: wrap_qid(x, CONFIG["color_wikidata_green"]),
            "Reference QID": lambda x: wrap_qid(x, CONFIG["color_wikidata_blue"]),
        },
        page_size=CONFIG["page_size_default"],
    )
    mo.vstack([
        mo.md("## Results"),
        stats_ui,
        table_ui,
    ])
    return


@app.cell
def is_large(lotus, results, stats, qid, taxon_input):
    """Generate download buttons."""
    if results is None or stats is None or stats.n_entries == 0:
        mo.Html("")

    is_large = stats.n_entries > CONFIG["table_row_limit"]
    return(is_large)

@app.cell
def generate_buttons(lotus, results, stats, qid, taxon_input):
    if is_large:
        # Large dataset - on-demand generation
        csv_btn = mo.ui.run_button(label="Generate CSV")
        json_btn = mo.ui.run_button(label="Generate JSON")
        rdf_btn = mo.ui.run_button(label="Generate RDF")

    # Prepare export data once
    export_df = lotus.prepare_export_dataframe(results, include_rdf_ref=False)
    rdf_df = lotus.prepare_export_dataframe(results, include_rdf_ref=True)

    # Delete results to free memory
    del results
    if IS_PYODIDE:
        gc.collect()
    return(csv_btn,json_btn,rdf_btn,export_df,rdf_df)


@app.cell
def generate_downloads(lotus, results, stats, qid, taxon_input):
    if is_large:
        # CSV
        if csv_btn.value:
            with mo.status.spinner("Generating CSV..."):
                csv_bytes = lotus.export(export_df, "csv")
                csv_ui = create_download_button(
                    csv_bytes,
                    generate_filename(taxon_input.value, "csv"),
                    "Download CSV",
                    "text/csv"
                )
        else:
            csv_ui = mo.Html("")

        # JSON
        if json_btn.value:
            with mo.status.spinner("Generating JSON..."):
                json_bytes = lotus.export(export_df, "json")
                json_ui = create_download_button(
                    json_bytes,
                    generate_filename(taxon_input.value, "json"),
                    "Download JSON",
                    "application/json"
                )
        else:
            json_ui = mo.Html("")

        # RDF
        if rdf_btn.value:
            with mo.status.spinner("Generating RDF..."):
                rdf_bytes = lotus.export(
                    rdf_df,
                    "rdf",
                    taxon_input=taxon_input.value,
                    qid=qid,
                    filters={}
                )
                rdf_ui = create_download_button(
                    rdf_bytes,
                    generate_filename(taxon_input.value, "ttl"),
                    "Download RDF",
                    "text/turtle"
                )
        else:
            rdf_ui = mo.Html("")

        _out = mo.vstack([
            mo.md("### Download Data"),
            mo.hstack([csv_btn, json_btn, rdf_btn], gap=2),
            csv_ui,
            json_ui,
            rdf_ui,
        ])
    else:
        # Small dataset - generate all immediately
        csv_bytes = lotus.export(export_df, "csv")
        json_bytes = lotus.export(export_df, "json")
        rdf_bytes = lotus.export(
            rdf_df,
            "rdf",
            taxon_input=taxon_input.value,
            qid=qid,
            filters={}
        )
        _out=mo.vstack([
            mo.md("### Download Data"),
            mo.hstack([
                create_download_button(csv_bytes, generate_filename(taxon_input.value, "csv"), "CSV", "text/csv"),
                create_download_button(json_bytes, generate_filename(taxon_input.value, "json"), "JSON", "application/json"),
                create_download_button(rdf_bytes, generate_filename(taxon_input.value, "ttl"), "RDF", "text/turtle"),
            ], gap=2)
        ])
    _out
    return


@app.cell
def footer():
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


if __name__ == "__main__":
    app.run()
