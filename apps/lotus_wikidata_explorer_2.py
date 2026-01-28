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
LOTUS Wikidata Explorer 2

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

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import polars as pl
    import io
    import json
    import time
    import hashlib
    import sys
    import urllib.parse
    import gc
    from dataclasses import dataclass
    from datetime import datetime
    from rdflib import Graph, Literal, URIRef, BNode
    from rdflib.namespace import RDF, RDFS, XSD, DCTERMS
    from abc import ABC, abstractmethod

    _USE_LOCAL = True
    if _USE_LOCAL:
        sys.path.insert(0, ".")

    from modules.text.formula.filters import FormulaFilters
    from modules.text.formula.create_filters import create_filters
    from modules.text.formula.serialize_filters import serialize_filters
    from modules.text.formula.match_filters import match_filters
    from modules.text.smiles.validate_and_escape import validate_and_escape
    from modules.knowledge.wikidata.entity.extract_from_url import extract_from_url
    from modules.knowledge.wikidata.url.constants import (
        ENTITY_PREFIX as WIKIDATA_ENTITY_PREFIX,
        STATEMENT_PREFIX as WIKIDATA_STATEMENT_PREFIX,
        WIKIDATA_HTTP_BASE,
        WIKI_PREFIX,
    )
    from modules.knowledge.wikidata.sparql.query_taxon_search import query_taxon_search
    from modules.knowledge.wikidata.sparql.query_compounds import (
        query_compounds_by_taxon,
        query_all_compounds,
    )
    from modules.knowledge.wikidata.sparql.query_sachem import query_sachem
    from modules.net.sparql.execute_with_retry import execute_with_retry
    from modules.net.sparql.parse_response import parse_sparql_response
    from modules.chem.cdk.depict.svg_from_smiles import svg_from_smiles
    from modules.knowledge.rdf.graph.add_literal import add_literal
    from modules.knowledge.rdf.namespace.wikidata import WIKIDATA_NAMESPACES
    from modules.text.formula.element_config import ELEMENT_DEFAULTS
    from modules.io.compress.if_large import compress_if_large

    IS_PYODIDE = "pyodide" in sys.modules
    if IS_PYODIDE:
        import pyodide_http

        pyodide_http.patch_all()

    CONFIG = {
        "app_version": "0.2.0",
        "app_name": "LOTUS Wikidata Explorer",
        "app_url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
        "qlever_endpoint": "https://qlever.dev/api/wikidata",
        "table_row_limit": 1_000,
        "download_embed_threshold_bytes": 500_000,
        "color_hyperlink": "#3377c4",
        "color_wikidata_blue": "#006699",
        "color_wikidata_green": "#339966",
        "color_wikidata_red": "#990000",
        "page_size_default": 10,
        "page_size_export": 25,
    }

    PLURAL_MAP = {"Entry": "Entries", "Taxon": "Taxa"}

    # ========================================================================
    # DOMAIN MODELS
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
        formula_filters: FormulaFilters | None = None

        def has_mass_filter(self) -> bool:
            return self.mass_range != (0.0, 2000.0)

        def has_year_filter(self) -> bool:
            return self.year_range != (1900, datetime.now().year)

        def to_filters_dict(self) -> dict:
            """Convert to filters dictionary."""
            filters = {}
            if self.smiles:
                filters["chemical_structure"] = {
                    "smiles": self.smiles,
                    "search_type": self.smiles_search_type,
                }
                if self.smiles_search_type == "similarity":
                    filters["chemical_structure"]["similarity_threshold"] = (
                        self.smiles_threshold
                    )
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

    @dataclass(frozen=True)
    class DatasetStats:
        """Dataset statistics."""

        n_compounds: int
        n_taxa: int
        n_references: int
        n_entries: int

        @classmethod
        def from_lazyframe(cls, df: pl.LazyFrame) -> "DatasetStats":
            stats = df.select(
                [
                    pl.col("compound").n_unique().cast(pl.UInt32).alias("n_compounds"),
                    pl.col("taxon").n_unique().cast(pl.UInt32).alias("n_taxa"),
                    pl.col("reference").n_unique().cast(pl.UInt32).alias("n_refs"),
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
        def __init__(self, is_wasm: bool):
            self.is_wasm = is_wasm

        def get_batch_size(self, format: str) -> int:
            sizes = {"csv": (2000, 10000), "json": (5000, 10000), "rdf": (500, 2000)}
            wasm_size, desktop_size = sizes.get(format, (1000, 5000))
            return wasm_size if self.is_wasm else desktop_size

    # ========================================================================
    # SERVICES
    # ========================================================================

    class WikidataQueryService:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def fetch_compounds(
            self,
            qid: str,
            smiles: str | None = None,
            smiles_search_type: str = "substructure",
            smiles_threshold: float = 0.8,
        ) -> pl.LazyFrame:
            query = self._build_query(qid, smiles, smiles_search_type, smiles_threshold)
            csv_bytes = execute_with_retry(query, self.endpoint)

            if not csv_bytes or csv_bytes.strip() == b"":
                return pl.LazyFrame()

            return pl.scan_csv(
                io.BytesIO(csv_bytes),
                low_memory=True,
                rechunk=False,
                dtypes={
                    "compound": pl.UInt32,
                    "taxon": pl.UInt32,
                    "reference": pl.UInt32,
                    "compound_mass": pl.Float32,
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
        @staticmethod
        def apply_standard_transforms(df: pl.LazyFrame) -> pl.LazyFrame:
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
            return df.with_columns(
                [
                    pl.coalesce(["compound_smiles_iso", "compound_smiles_conn"]).alias(
                        "smiles",
                    ),
                ],
            )

        @staticmethod
        def extract_doi(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(
                [
                    pl.when(pl.col("ref_doi").str.starts_with("http"))
                    .then(pl.col("ref_doi").str.split("doi.org/").list.last())
                    .otherwise(pl.col("ref_doi"))
                    .alias("ref_doi"),
                ],
            )

        @staticmethod
        def parse_dates(df: pl.LazyFrame) -> pl.LazyFrame:
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
            return df.with_columns(
                [
                    pl.col("compound").cast(pl.UInt32),
                    pl.col("name").cast(pl.Categorical),
                    pl.col("inchikey").cast(pl.Categorical),
                    pl.col("smiles").cast(pl.Categorical),
                    pl.col("taxon_name").cast(pl.Categorical),
                    pl.col("taxon").cast(pl.UInt32),
                    pl.col("ref_title").cast(pl.Categorical),
                    pl.col("ref_doi").cast(pl.Categorical),
                    pl.col("reference").cast(pl.UInt32),
                    pl.col("mass").cast(pl.Float32),
                    pl.col("mf").cast(pl.Categorical),
                    pl.col("statement").cast(pl.Categorical),
                    pl.col("ref").cast(pl.Categorical),
                ]
            )

        @staticmethod
        def drop_old_columns(df: pl.LazyFrame) -> pl.LazyFrame:
            to_drop = ["compound_smiles_iso", "compound_smiles_conn"]
            return df.drop(
                [col for col in to_drop if col in df.collect_schema().names()],
            )

        @staticmethod
        def add_missing_columns(df: pl.LazyFrame) -> pl.LazyFrame:
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
            return df.unique(
                subset=["compound", "taxon", "reference"],
                keep="first",
            ).sort("name")

    class FilterService:
        @staticmethod
        def apply_filters(df: pl.LazyFrame, criteria: SearchCriteria) -> pl.LazyFrame:
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
            return df.filter(
                pl.col("mf").map_batches(
                    lambda s: s.map_elements(
                        lambda f: match_filters(f or "", formula_filters),
                        return_dtype=pl.Boolean,
                    ),
                ),
            )

    class TaxonResolutionService:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def resolve(self, taxon_input: str) -> tuple[str | None, mo.Html | None]:
            taxon_input = str(taxon_input).strip()

            if not taxon_input:
                return None, None
            if taxon_input == "*":
                return "*", None
            if taxon_input.upper().startswith("Q") and taxon_input[1:].isdigit():
                return taxon_input.upper(), None

            try:
                query = query_taxon_search(taxon_input)
                csv_bytes = execute_with_retry(
                    query,
                    self.endpoint,
                    fallback_endpoint=None,
                )

                if not csv_bytes or not csv_bytes.strip():
                    return None, None

                df = parse_sparql_response(csv_bytes).collect()
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
                    return exact_matches[0][0], None

                return matches[0][0], None

            except Exception:
                return None, None

    # ========================================================================
    # EXPORT STRATEGIES
    # ========================================================================

    class ExportStrategy(ABC):
        def __init__(self, memory: MemoryManager):
            self.memory = memory

        def export(self, df: pl.LazyFrame) -> bytes:
            return self._to_bytes(df)

        @abstractmethod
        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            pass

    class CSVExportStrategy(ExportStrategy):
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
        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            df_collected = df.collect()
            json_str = df_collected.write_json()
            result = json_str.encode("utf-8")
            del df_collected
            if self.memory.is_wasm:
                gc.collect()
            return result

    class RDFExportStrategy(ExportStrategy):
        def __init__(
            self,
            memory: MemoryManager,
            taxon_input: str,
            qid: str,
            filters: dict,
        ):
            super().__init__(memory)
            self.taxon_input = taxon_input
            self.qid = qid
            self.filters = filters

        def _to_bytes(self, df: pl.LazyFrame) -> bytes:
            df_collected = df.collect()

            g = Graph()
            for prefix, ns in WIKIDATA_NAMESPACES.items():
                g.bind(prefix.lower(), ns)
            g.bind("rdfs", RDFS)
            g.bind("xsd", XSD)
            g.bind("dcterms", DCTERMS)

            dataset_uri, query_hash, result_hash = self._create_dataset_uri(
                df_collected,
            )
            self._add_metadata(
                g,
                dataset_uri,
                len(df_collected),
                query_hash,
                result_hash,
            )

            batch_size = self.memory.get_batch_size("rdf")
            processed_taxa = set()
            processed_refs = set()
            ns_cache = {
                k: WIKIDATA_NAMESPACES[k]
                for k in ["WD", "WDT", "P", "PS", "PR", "PROV", "SCHEMA"]
            }

            for start_idx in range(0, len(df_collected), batch_size):
                batch = df_collected[start_idx : start_idx + batch_size]
                for row in batch.iter_rows(named=True):
                    self._add_compound_triples(
                        g,
                        row,
                        dataset_uri,
                        processed_taxa,
                        processed_refs,
                        ns_cache,
                    )
                del batch

            result = g.serialize(format="turtle").encode("utf-8")
            del df_collected, g
            return result

        def _create_dataset_uri(self, df: pl.DataFrame) -> tuple[URIRef, str, str]:
            # Query hash
            query_components = [self.qid or "", self.taxon_input or ""]
            if self.filters:
                query_components.append(json.dumps(self.filters, sort_keys=True))
            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8"),
            ).hexdigest()

            # Result hash
            result_hasher = hashlib.sha256()
            for val in (
                df.select(pl.col("compound_qid"))
                .to_series()
                .drop_nulls()
                .unique()
                .sort()
            ):
                result_hasher.update(str(val).encode("utf-8"))
            result_hash = result_hasher.hexdigest()

            return URIRef(f"urn:hash:sha256:{result_hash}"), query_hash, result_hash

        def _add_metadata(
            self,
            g: Graph,
            dataset_uri: URIRef,
            record_count: int,
            query_hash: str,
            result_hash: str,
        ):
            SCHEMA = WIKIDATA_NAMESPACES["SCHEMA"]
            WD = WIKIDATA_NAMESPACES["WD"]

            g.add((dataset_uri, RDF.type, SCHEMA.Dataset))
            g.add(
                (dataset_uri, SCHEMA.name, Literal(f"LOTUS Data - {self.taxon_input}")),
            )
            g.add(
                (
                    dataset_uri,
                    SCHEMA.numberOfRecords,
                    Literal(record_count, datatype=XSD.integer),
                ),
            )
            g.add((dataset_uri, SCHEMA.version, Literal(CONFIG["app_version"])))
            g.add((dataset_uri, SCHEMA.provider, URIRef(CONFIG["app_url"])))
            g.add((dataset_uri, DCTERMS.source, URIRef(WIKIDATA_HTTP_BASE)))
            g.add((dataset_uri, SCHEMA.isBasedOn, URIRef(WIKI_PREFIX + "Q104225190")))

            if self.qid and self.qid != "*":
                g.add((dataset_uri, SCHEMA.about, WD[self.qid]))

            g.add(
                (dataset_uri, DCTERMS.provenance, Literal(f"Query hash: {query_hash}")),
            )
            g.add((dataset_uri, DCTERMS.identifier, Literal(f"sha256:{result_hash}")))

        def _add_compound_triples(
            self,
            g: Graph,
            row: dict,
            dataset_uri: URIRef,
            processed_taxa: set,
            processed_refs: set,
            ns_cache: dict,
        ):
            WD, WDT, P, PS, PR, PROV, SCHEMA = (
                ns_cache[k] for k in ["WD", "WDT", "P", "PS", "PR", "PROV", "SCHEMA"]
            )

            compound_qid = row.get("compound_qid")
            if not compound_qid:
                return

            compound_uri = WD[compound_qid]
            g.add((dataset_uri, SCHEMA.hasPart, compound_uri))

            add_literal(g, compound_uri, WDT.P235, row.get("compound_inchikey"))
            add_literal(g, compound_uri, WDT.P233, row.get("compound_smiles"))
            add_literal(g, compound_uri, WDT.P274, row.get("molecular_formula"))
            add_literal(g, compound_uri, RDFS.label, row.get("compound_name"))

            if row.get("compound_mass") is not None:
                add_literal(g, compound_uri, WDT.P2067, row["compound_mass"], XSD.float)

            # Full statement structure with provenance
            taxon_qid = row.get("taxon_qid")
            ref_qid = row.get("reference_qid")
            statement_uri_str = row.get("statement_id")
            ref_uri_str = row.get("ref")

            if taxon_qid:
                taxon_uri = WD[taxon_qid]
                statement_node = (
                    URIRef(WIKIDATA_STATEMENT_PREFIX + statement_uri_str)
                    if statement_uri_str
                    else BNode()
                )

                g.add((compound_uri, P.P703, statement_node))
                g.add((statement_node, PS.P703, taxon_uri))

                if ref_qid:
                    ref_uri = WD[ref_qid]
                    ref_node = URIRef(ref_uri_str) if ref_uri_str else BNode()

                    g.add((statement_node, PROV.wasDerivedFrom, ref_node))
                    g.add((ref_node, PR.P248, ref_uri))

                    if ref_qid not in processed_refs:
                        add_literal(g, ref_uri, WDT.P1476, row.get("reference_title"))
                        add_literal(g, ref_uri, RDFS.label, row.get("reference_title"))
                        add_literal(g, ref_uri, WDT.P356, row.get("reference_doi"))
                        if row.get("reference_date"):
                            add_literal(
                                g,
                                ref_uri,
                                WDT.P577,
                                str(row["reference_date"]),
                                XSD.date,
                            )
                        processed_refs.add(ref_qid)

                g.add((compound_uri, WDT.P703, taxon_uri))

                if taxon_qid not in processed_taxa:
                    add_literal(g, taxon_uri, WDT.P225, row.get("taxon_name"))
                    add_literal(g, taxon_uri, RDFS.label, row.get("taxon_name"))
                    processed_taxa.add(taxon_qid)

    # ========================================================================
    # FACADE
    # ========================================================================

    class LOTUSExplorer:
        def __init__(self, config: dict, is_wasm: bool = False):
            self.config = config
            self.memory = MemoryManager(is_wasm)
            self.query_service = WikidataQueryService(config["qlever_endpoint"])
            self.transform_service = DataTransformService()
            self.filter_service = FilterService()
            self.taxon_service = TaxonResolutionService(config["qlever_endpoint"])

        def resolve_taxon(
            self,
            taxon_input: str,
        ) -> tuple[str | None, mo.Html | None]:
            return self.taxon_service.resolve(taxon_input)

        def search(
            self,
            criteria: SearchCriteria,
            qid: str,
        ) -> tuple[pl.LazyFrame, DatasetStats]:
            raw_data = self.query_service.fetch_compounds(
                qid,
                criteria.smiles,
                criteria.smiles_search_type,
                criteria.smiles_threshold,
            )
            transformed_data = self.transform_service.apply_standard_transforms(
                raw_data,
            )
            filtered_data = self.filter_service.apply_filters(
                transformed_data,
                criteria,
            )
            stats = DatasetStats.from_lazyframe(filtered_data)
            return filtered_data, stats

        def export(self, df: pl.LazyFrame, format: str, **kwargs) -> bytes:
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

        def prepare_export_dataframe(
            self,
            df: pl.LazyFrame,
            include_rdf_ref: bool = False,
        ) -> pl.LazyFrame:
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
                pl.concat_str([pl.lit("Q"), pl.col("compound")]).alias(
                    "compound_qid",
                ),
                pl.concat_str([pl.lit("Q"), pl.col("taxon")]).alias(
                    "taxon_qid",
                ),
                pl.concat_str([pl.lit("Q"), pl.col("reference")]).alias(
                    "reference_qid",
                ),
            ]

            if "statement" in df.collect_schema().names():
                exprs.append(
                    pl.col("statement")
                    .str.replace(WIKIDATA_STATEMENT_PREFIX, "", literal=True)
                    .alias("statement_id"),
                )

            if include_rdf_ref and "ref" in df.collect_schema().names():
                exprs.append(pl.col("ref"))

            return df.select(exprs)

        def build_display_dataframe(self, df: pl.LazyFrame, limit: int) -> pl.DataFrame:
            df = (
                df.limit(limit).collect()
                if limit
                else df.collect()
            )
            return df.select(
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

        def compute_hashes(
            self,
            qid: str,
            taxon_input: str,
            filters: dict,
            df: pl.LazyFrame,
        ) -> tuple[str, str]:
            # Query hash
            query_components = [qid or "", taxon_input or ""]
            if filters:
                query_components.append(json.dumps(filters, sort_keys=True))
            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8"),
            ).hexdigest()

            # Result hash
            result_hasher = hashlib.sha256()
            try:
                df_temp = (
                    df.select(pl.col("compound"))
                    .drop_nulls()
                    .unique()
                    .sort("compound")
                    .collect()
                )
                for val in df_temp.get_column("compound").to_list():
                    if val:
                        result_hasher.update(str(val).encode("utf-8"))
                del df_temp
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
            smiles_info = filters.get("chemical_structure", {})
            if smiles_info:
                search_type = smiles_info.get("search_type", "substructure")
                dataset_name = (
                    f"LOTUS Data - {search_type.title()} search in {taxon_input}"
                )
            else:
                dataset_name = f"LOTUS Data - {taxon_input}"

            metadata = {
                "@context": "https://schema.org/",
                "@type": "Dataset",
                "name": dataset_name,
                "version": CONFIG["app_version"],
                "dateCreated": datetime.now().isoformat(),
                "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                "creator": {
                    "@type": "SoftwareApplication",
                    "name": CONFIG["app_name"],
                    "version": CONFIG["app_version"],
                    "url": CONFIG["app_url"],
                },
                "numberOfRecords": stats.n_entries,
                "search_parameters": {"taxon": taxon_input, "taxon_qid": qid},
            }

            if filters:
                metadata["search_parameters"]["filters"] = filters

            metadata["provenance"] = {
                "query_hash": {"algorithm": "SHA-256", "value": query_hash},
                "result_hash": {"algorithm": "SHA-256", "value": result_hash},
                "dataset_uri": f"urn:hash:sha256:{result_hash}",
            }

            return metadata

        def create_citation(self, taxon_input: str) -> str:
            current_date = datetime.now().strftime("%B %d, %Y")
            return f"""
    ## How to Cite This Data

    ### Dataset Citation
    LOTUS Initiative via Wikidata. ({datetime.now().year}). *Data for {taxon_input}*.
    Retrieved from LOTUS Wikidata Explorer on {current_date}.
    License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

    ### LOTUS Initiative Publication
    Rutz A, Sorokina M, Galgonek J, et al. (2022). The LOTUS initiative for open knowledge
    management in natural products research. *eLife* **11**:e70780.
    DOI: [10.7554/eLife.70780](https://doi.org/10.7554/eLife.70780)
    """

        def build_shareable_url(self, criteria: SearchCriteria) -> str:
            params = {}
            if criteria.taxon:
                params["taxon"] = criteria.taxon
            if criteria.smiles:
                params["smiles"] = criteria.smiles
                params["smiles_search_type"] = criteria.smiles_search_type
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

    def generate_filename(taxon_name: str, file_type: str, filters: dict = None) -> str:
        safe_name = (
            taxon_name.replace(" ", "_").replace("/", "_")
            if taxon_name != "*"
            else "all_taxa"
        )
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{date_str}_lotus_{safe_name}.{file_type}"

    def create_download_button(data: bytes, filename: str, label: str, mimetype: str):
        compressed, was_compressed = compress_if_large(
            data,
            CONFIG["download_embed_threshold_bytes"],
        )
        final_filename = filename + ".gz" if was_compressed else filename
        final_mimetype = "application/gzip" if was_compressed else mimetype
        display_label = label + (" (gzipped)" if was_compressed else "")
        return mo.download(
            data=compressed,
            filename=final_filename,
            label=display_label,
            mimetype=final_mimetype,
        )


@app.cell
def md_title():
    mo.md("""
    # LOTUS Wikidata Explorer
    """)
    return


@app.cell
def ui_help():
    mo.accordion(
        {
            "Help & API": mo.md("""
    **Search:** Enter a taxon name (e.g., *Gentiana lutea*) and/or a SMILES structure.

    **URL API:** `?taxon=Salix&smiles=CC(=O)Oc1ccccc1C(=O)O`
    """),
        },
    )
    return


@app.cell
def ui_search_inputs():
    taxon_input = mo.ui.text(
        value="Gentiana lutea",
        label="Taxon Name or QID",
        full_width=True,
    )
    smiles_input = mo.ui.text(value="", label="SMILES", full_width=True)
    smiles_search_type = mo.ui.dropdown(
        options=["substructure", "similarity"],
        value="substructure",
    )
    smiles_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.8,
        label="Threshold",
    )
    mass_filter = mo.ui.checkbox(label="Mass filter", value=False)
    mass_min = mo.ui.number(value=0, start=0, stop=10000, label="Min (Da)")
    mass_max = mo.ui.number(value=2000, start=0, stop=10000, label="Max (Da)")
    year_filter = mo.ui.checkbox(label="Year filter", value=False)
    year_start = mo.ui.number(value=1900, start=1700, stop=2024, label="From")
    year_end = mo.ui.number(value=2024, start=1700, stop=2024, label="To")
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

    run_button = mo.ui.run_button(label="Search Wikidata")
    return (
        c_max,
        c_min,
        exact_formula,
        formula_filter,
        h_max,
        h_min,
        mass_filter,
        mass_max,
        mass_min,
        n_max,
        n_min,
        o_max,
        o_min,
        run_button,
        smiles_input,
        smiles_search_type,
        smiles_threshold,
        taxon_input,
        year_end,
        year_filter,
        year_start,
    )


@app.cell
def ui_search_panel(
    c_max,
    c_min,
    exact_formula,
    formula_filter,
    h_max,
    h_min,
    mass_filter,
    mass_max,
    mass_min,
    n_max,
    n_min,
    o_max,
    o_min,
    run_button,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    structure_fields = [smiles_input, smiles_search_type]
    if smiles_search_type.value == "similarity":
        structure_fields.append(smiles_threshold)

    main_search = mo.hstack(
        [mo.vstack([taxon_input, run_button]), mo.vstack(structure_fields)],
        gap=2,
        widths="equal",
    )

    filter_row = mo.hstack([mass_filter, year_filter, formula_filter], gap=2)
    filters_ui = [filter_row, main_search]

    if mass_filter.value:
        filters_ui.append(mo.hstack([mass_min, mass_max], gap=2))
    if year_filter.value:
        filters_ui.append(mo.hstack([year_start, year_end], gap=2))
    if formula_filter.value:
        filters_ui.extend(
            [
                exact_formula,
                mo.hstack([c_min, c_max, h_min, h_max], gap=1),
                mo.hstack([n_min, n_max, o_min, o_max], gap=1),
            ],
        )

    mo.vstack(filters_ui, gap=1)
    return


@app.cell
def execute_search(
    c_max,
    c_min,
    exact_formula,
    formula_filter,
    h_max,
    h_min,
    mass_filter,
    mass_max,
    mass_min,
    n_max,
    n_min,
    o_max,
    o_min,
    run_button,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    if not run_button.value:
        lotus, results, stats, qid, criteria, query_hash, result_hash = (
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

        qid, taxon_warning = lotus.resolve_taxon(taxon_input.value)
        if not qid:
            mo.stop(
                True,
                mo.callout(
                    mo.md(f"**Taxon not found:** {taxon_input.value}"),
                    kind="warn",
                ),
            )

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
                p_min=0,
                p_max=ELEMENT_DEFAULTS["p"],
                s_min=0,
                s_max=ELEMENT_DEFAULTS["s"],
                f_state="allowed",
                cl_state="allowed",
                br_state="allowed",
                i_state="allowed",
            )

        criteria = SearchCriteria(
            taxon=taxon_input.value,
            smiles=smiles_input.value.strip() if smiles_input.value else "",
            smiles_search_type=smiles_search_type.value,
            smiles_threshold=smiles_threshold.value,
            mass_range=(mass_min.value, mass_max.value)
            if mass_filter.value
            else (0.0, 2000.0),
            year_range=(year_start.value, year_end.value)
            if year_filter.value
            else (1900, 2024),
            formula_filters=formula_filt,
        )

        with mo.status.spinner("Searching..."):
            results, stats = lotus.search(criteria, qid)

        query_hash, result_hash = lotus.compute_hashes(
            qid,
            taxon_input.value,
            criteria.to_filters_dict(),
            results,
        )

        elapsed = round(time.time() - start_time, 2)
        mo.md(f"Query executed in **{elapsed}s**")
    return criteria, lotus, qid, query_hash, result_hash, results, stats


@app.cell
def display_results(
    criteria,
    lotus,
    qid,
    query_hash,
    result_hash,
    results,
    stats,
    taxon_input,
):
    if results is None or stats is None:
        result_ui = mo.Html("")
    elif stats.n_entries == 0:
        result_ui = mo.callout(
            mo.md(f"No compounds found for **{taxon_input.value}**"),
            kind="warn",
        )
    else:
        display_df = lotus.build_display_dataframe(results, CONFIG["table_row_limit"])

        def wrap_image2(smiles: str):
            return mo.image(svg_from_smiles(smiles)) if smiles else ""

        def wrap_qid(qid_val: str, color: str):
            if not qid_val:
                return ""
            url = f"https://scholia.toolforge.org/Q{qid_val}"
            return mo.Html(
                f'<a href="{url}" style="color:{color};" target="_blank">Q{qid_val}</a>',
            )

        def wrap_doi(doi: str):
            if not doi:
                return ""
            url = f"https://doi.org/{doi}"
            return mo.Html(
                f'<a href="{url}" style="color:{CONFIG["color_hyperlink"]};" target="_blank">{doi}</a>',
            )

        def wrap_statement(statement: str):
            if not statement:
                return ""
            statement_id = statement.replace(WIKIDATA_STATEMENT_PREFIX, "")
            url = f"https://www.wikidata.org/entity/statement/{statement_id}"
            return mo.Html(
                f'<a href="{url}" style="color:{CONFIG["color_hyperlink"]};" target="_blank">{statement_id}</a>',
            )

        shareable_url = lotus.build_shareable_url(criteria)
        search_info = mo.md(
            f"**{taxon_input.value}** ([{qid}](https://scholia.toolforge.org/{qid}))\n\n"
            f"**Hashes:** *Query*: `{query_hash}` | *Results*: `{result_hash}`",
        )

        stats_ui = mo.hstack(
            [
                mo.stat(value=f"{stats.n_compounds:,}", label="Compounds"),
                mo.stat(value=f"{stats.n_taxa:,}", label="Taxa"),
                mo.stat(value=f"{stats.n_references:,}", label="References"),
                mo.stat(value=f"{stats.n_entries:,}", label="Entries"),
            ],
            gap=0,
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
            .limit(100)
            .collect()
        )
        metadata = lotus.create_metadata(
            stats,
            taxon_input.value,
            qid,
            criteria.to_filters_dict(),
            query_hash,
            result_hash,
        )
        citation = lotus.create_citation(taxon_input.value)

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

        if shareable_url:
            url_accordion = mo.accordion(
                {
                    "Share this search": mo.md(
                        f"Copy and append to notebook URL:\n```\n{shareable_url}\n```",
                    ),
                },
            )
            result_ui = mo.vstack(
                [mo.md("## Results"), search_info, stats_ui, url_accordion, tabs_ui],
            )
        else:
            result_ui = mo.vstack([mo.md("## Results"), search_info, stats_ui, tabs_ui])

    result_ui
    return


@app.cell
def is_large(results, stats):
    """Generate download buttons."""
    if results is None or stats is None or stats.n_entries == 0:
        mo.Html("")

    is_large = stats.n_entries > CONFIG["table_row_limit"]
    return (is_large,)


@app.cell
def generate_buttons(is_large, lotus, results):
    if is_large:
        # Large dataset - on-demand generation
        csv_btn = mo.ui.run_button(label="Generate CSV")
        json_btn = mo.ui.run_button(label="Generate JSON")
        rdf_btn = mo.ui.run_button(label="Generate RDF")

    # Prepare export data once
    rdf_df = lotus.prepare_export_dataframe(results, include_rdf_ref=True)

    # Delete results to free memory
    del results
    if IS_PYODIDE:
        gc.collect()
    return csv_btn, json_btn, rdf_btn, rdf_df


@app.cell
def generate_downloads(
    criteria,
    csv_btn,
    is_large,
    json_btn,
    lotus,
    qid,
    rdf_btn,
    rdf_df,
    taxon_input,
):
    if is_large:
        # CSV
        if csv_btn.value:
            with mo.status.spinner("Generating CSV..."):
                csv_bytes = lotus.export(rdf_df.drop("ref"), "csv")
                csv_ui = create_download_button(
                    csv_bytes,
                    generate_filename(taxon_input.value, "csv"),
                    "CSV",
                    "text/csv",
                )
        else:
            csv_ui = mo.Html("")

        if json_btn.value:
            with mo.status.spinner("Generating JSON..."):
                json_bytes = lotus.export(rdf_df.drop("ref"), "json")
                json_ui = create_download_button(
                    json_bytes,
                    generate_filename(taxon_input.value, "json"),
                    "JSON",
                    "application/json",
                )
        else:
            json_ui = mo.Html("")

        if rdf_btn.value:
            with mo.status.spinner("Generating RDF..."):
                rdf_bytes = lotus.export(
                    rdf_df,
                    "rdf",
                    taxon_input=taxon_input.value,
                    qid=qid,
                    filters=criteria.to_filters_dict(),
                )
                rdf_ui = create_download_button(
                    rdf_bytes,
                    generate_filename(taxon_input.value, "ttl"),
                    "RDF",
                    "text/turtle",
                )
        else:
            rdf_ui = mo.Html("")

        download_ui = mo.vstack(
            [
                mo.md("### Download Data"),
                mo.hstack([csv_btn, json_btn, rdf_btn], gap=2),
                csv_ui,
                json_ui,
                rdf_ui,
            ],
        )
    else:
        csv_bytes = lotus.export(rdf_df.drop("ref"), "csv")
        json_bytes = lotus.export(rdf_df.drop("ref"), "json")
        rdf_bytes = lotus.export(
            rdf_df,
            "rdf",
            taxon_input=taxon_input.value,
            qid=qid,
            filters=criteria.to_filters_dict(),
        )

        download_ui = mo.vstack(
            [
                mo.md("### Download Data"),
                mo.hstack(
                    [
                        create_download_button(
                            csv_bytes,
                            generate_filename(taxon_input.value, "csv"),
                            "CSV",
                            "text/csv",
                        ),
                        create_download_button(
                            json_bytes,
                            generate_filename(taxon_input.value, "json"),
                            "JSON",
                            "application/json",
                        ),
                        create_download_button(
                            rdf_bytes,
                            generate_filename(taxon_input.value, "ttl"),
                            "RDF",
                            "text/turtle",
                        ),
                    ],
                    gap=2,
                ),
            ],
        )

    download_ui
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
