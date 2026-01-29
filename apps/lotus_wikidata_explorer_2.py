# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "great-tables==0.20.0",
#     "marimo",
#     "polars==1.37.1",
#     "rdflib==7.5.0",
# ]
# [tool.marimo.runtime]
# output_max_bytes = 100_000_000
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
    from datetime import date, datetime
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
    from modules.knowledge.rdf.graph.add_literal import add_literal
    from modules.knowledge.rdf.namespace.wikidata import WIKIDATA_NAMESPACES
    from modules.io.compress.if_large import compress_if_large

    IS_PYODIDE = "pyodide" in sys.modules
    if IS_PYODIDE:
        import pyodide_http

        pyodide_http.patch_all()

    CONFIG = {
        "app_version": "0.1.0",
        "app_name": "LOTUS Wikidata Explorer",
        "app_url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
        "qlever_endpoint": "https://qlever.dev/api/wikidata",
        "table_row_limit": 500,
        "download_embed_threshold_bytes": 500_000,
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

    @dataclass(frozen=True, slots=True)
    class DatasetStats:
        """Dataset statistics."""

        n_compounds: int = 0
        n_taxa: int = 0
        n_references: int = 0
        n_entries: int = 0

        @classmethod
        def from_lazyframe(cls, df: pl.LazyFrame) -> "DatasetStats":
            stats = df.select(
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
            )

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

                if len(exact_matches) > 1:
                    return self._resolve_ambiguous(exact_matches, is_exact=True)

                if len(matches) > 1:
                    return self._resolve_ambiguous(matches, is_exact=False)

                return matches[0][0], None

            except Exception:
                return None, None

        def _resolve_ambiguous(
            self,
            matches: list[tuple[str, str]],
            is_exact: bool,
        ) -> tuple[str, mo.Html]:
            """Resolve ambiguous taxon matches."""
            qids = tuple(qid for qid, _ in matches)
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
                        info[qid][1] = row.get("taxonDescription", "")
                        info[qid][2] = row.get("taxon_parentLabel", "")

            selected_qid = max(qids, key=lambda q: info[q][0])

            matches_sorted = sorted(matches, key=lambda x: info[x[0]][0], reverse=True)
            matches_with_details = [
                (qid, name, info[qid][1], info[qid][2], info[qid][0])
                for qid, name in matches_sorted
            ]

            return selected_qid, self._create_taxon_warning_html(
                matches_with_details,
                selected_qid,
                is_exact,
            )

        def _create_taxon_warning_html(
            self,
            matches: list,
            selected_qid: str,
            is_exact: bool,
        ) -> mo.Html:
            """Create HTML warning for ambiguous taxon."""
            match_type = "exact matches" if is_exact else "similar taxa"
            intro = (
                f"Ambiguous taxon name. Multiple {match_type} found:"
                if is_exact
                else "No exact match. Similar taxa found:"
            )

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
                if self.memory.is_wasm:
                    gc.collect()

            result = g.serialize(format="turtle").encode("utf-8")
            del df_collected, g
            if self.memory.is_wasm:
                gc.collect()
            return result

        def _create_dataset_uri(self, df: pl.DataFrame) -> tuple[URIRef, str, str]:
            query_components = [self.qid or "", self.taxon_input or ""]
            if self.filters:
                query_components.append(json.dumps(self.filters, sort_keys=True))
            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8"),
            ).hexdigest()

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
            query_components = [qid or "", taxon_input or ""]
            if filters:
                query_components.append(json.dumps(filters, sort_keys=True))
            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8"),
            ).hexdigest()

            result_hasher = hashlib.sha256()
            try:
                unique_compounds = (
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
            smiles_info = filters.get("chemical_structure", {})
            if smiles_info:
                search_type = smiles_info.get("search_type", "substructure")
                dataset_name = (
                    f"LOTUS Data - {search_type.title()} search in {taxon_input}"
                )
                description = f"Chemical compounds from taxon {taxon_input}"
                if qid and qid != "*":
                    description += f" (Wikidata QID: {qid})"
                description += f". Retrieved via LOTUS Wikidata Explorer with {search_type} chemical search (SACHEM/IDSM)."
            else:
                dataset_name = f"LOTUS Data - {taxon_input}"
                description = f"Chemical compounds from taxon {taxon_input}"
                if qid and qid != "*":
                    description += f" (Wikidata QID: {qid})"
                description += ". Retrieved via LOTUS Wikidata Explorer."

            metadata = {
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
                    "taxon": taxon_input,
                    "taxon_qid": qid if qid else None,
                },
            }

            if smiles_info:
                metadata["provider"].append(
                    {
                        "@type": "Organization",
                        "name": "IDSM",
                        "url": "https://idsm.elixir-czech.cz/",
                    }
                )
                metadata["chemical_search_service"] = {
                    "name": "SACHEM",
                    "provider": "IDSM",
                    "endpoint": "https://idsm.elixir-czech.cz/sparql/endpoint/",
                }

            if filters:
                metadata["search_parameters"]["filters"] = filters

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


# @app.cell
# def ui_disclaimer():
#     mo.callout(
#         mo.md(
#             """
#             To run this script locally:

#             ```
#             uvx marimo run https://adafede.github.io/marimo/apps/lotus_wikidata_explorer.py
#             ```

#             """,
#         ),
#         kind="info",
#     ).style(
#         style={
#             "overflow-wrap": "anywhere",
#         },
#     )
#     return

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
            smiles=smiles_input.value.strip() if smiles_input.value else "",
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
            results, stats = lotus.search(criteria, qid)

        query_hash, result_hash = lotus.compute_hashes(
            qid,
            taxon_input.value,
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
    taxon_input,
    taxon_warning,
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
            if qid == "*":
                url = "https://qlever.scholia.wiki/taxon/all"
            else:
                url = f"https://scholia.toolforge.org/{qid}"
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

        display_taxon = "all taxa" if taxon_input.value == "*" else taxon_input.value
        if qid == "*":
            search_info = mo.md(
                f"**{display_taxon}**\n\n"
                f"**Hashes:** *Query*: `{query_hash}` | *Results*: `{result_hash}`",
            )
        else:
            search_info = mo.md(
                f"**{display_taxon}** ([{qid}](https://scholia.toolforge.org/{qid}))\n\n"
                f"**Hashes:** *Query*: `{query_hash}` | *Results*: `{result_hash}`",
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

        shareable_url = lotus.build_shareable_url(criteria)

        result_parts = [mo.md("## Results"), search_info, stats_ui]
        if shareable_url:
            result_parts.append(
                mo.accordion(
                    {"Share this search": mo.md(f"```\n{shareable_url}\n```")},
                ),
            )
        if taxon_warning:
            result_parts.append(mo.callout(taxon_warning, kind="warn"))
        result_parts.append(tabs_ui)

        result_ui = mo.vstack(result_parts)

    result_ui
    return


@app.cell
def is_large(results, stats):
    """Check if dataset is large."""
    if results is None or stats is None:
        is_large = False
    else:
        is_large = stats.n_entries > CONFIG["table_row_limit"]
    return (is_large,)


@app.cell
def generate_buttons(is_large, lotus, results):
    if results is None or not is_large:
        csv_btn = None
        json_btn = None
        rdf_btn = None
        export_df = pl.LazyFrame(schema={"compound_qid": pl.Utf8})
        rdf_df = pl.LazyFrame(schema={"compound_qid": pl.Utf8})
    else:
        csv_btn = mo.ui.run_button(label="Generate CSV")
        json_btn = mo.ui.run_button(label="Generate JSON")
        rdf_btn = mo.ui.run_button(label="Generate RDF")
        export_df = lotus.prepare_export_dataframe(results, include_rdf_ref=False)
        rdf_df = lotus.prepare_export_dataframe(results, include_rdf_ref=True)

        del results
        if IS_PYODIDE:
            gc.collect()
    return csv_btn, export_df, json_btn, rdf_btn, rdf_df


@app.cell
def generate_downloads(
    criteria,
    csv_btn,
    export_df,
    is_large,
    json_btn,
    lotus,
    qid,
    rdf_btn,
    rdf_df,
    stats,
    taxon_input,
):
    if stats is None or stats.n_entries == 0:
        download_ui = mo.Html("")
    elif is_large:
        if csv_btn and csv_btn.value:
            with mo.status.spinner("Generating CSV..."):
                csv_bytes = lotus.export(export_df, "csv")
                csv_ui = create_download_button(
                    csv_bytes,
                    generate_filename(taxon_input.value, "csv"),
                    "CSV",
                    "text/csv",
                )
                del csv_bytes
                if IS_PYODIDE:
                    gc.collect()
        else:
            csv_ui = mo.Html("")

        if json_btn and json_btn.value:
            with mo.status.spinner("Generating JSON..."):
                json_bytes = lotus.export(export_df, "json")
                json_ui = create_download_button(
                    json_bytes,
                    generate_filename(taxon_input.value, "json"),
                    "JSON",
                    "application/json",
                )
                del json_bytes
                if IS_PYODIDE:
                    gc.collect()
        else:
            json_ui = mo.Html("")

        if rdf_btn and rdf_btn.value:
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
                del rdf_bytes
                if IS_PYODIDE:
                    gc.collect()
        else:
            rdf_ui = mo.Html("")

        download_ui = mo.vstack(
            [
                mo.md("### Download Data"),
                mo.hstack([csv_btn, json_btn, rdf_btn], gap=2, wrap=True),
                csv_ui,
                json_ui,
                rdf_ui,
            ],
        )
    else:
        csv_bytes = lotus.export(export_df, "csv")
        json_bytes = lotus.export(export_df, "json")
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
                    wrap=True,
                ),
            ],
        )

        # Clean up all bytes
        del csv_bytes, json_bytes, rdf_bytes
        if IS_PYODIDE:
            gc.collect()

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
        parser.add_argument("--smiles", help="SMILES string")
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
        parser.add_argument("--compress", action="store_true")
        parser.add_argument("--show-metadata", action="store_true")
        parser.add_argument("--export-metadata", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        args = parser.parse_args()

        try:
            lotus = LOTUSExplorer(CONFIG, False)

            if args.taxon is None:
                args.taxon = "*"

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
                ],
            ):
                formula_filt = create_filters(
                    exact_formula=args.formula or "",
                    c_min=args.c_min or 0,
                    c_max=args.c_max or ELEMENT_DEFAULTS["c"],
                    h_min=args.h_min or 0,
                    h_max=args.h_max or ELEMENT_DEFAULTS["h"],
                    n_min=args.n_min or 0,
                    n_max=args.n_max or ELEMENT_DEFAULTS["n"],
                    o_min=args.o_min or 0,
                    o_max=args.o_max or ELEMENT_DEFAULTS["o"],
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
                taxon=args.taxon,
                smiles=args.smiles or "",
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
                sys.exit(0)

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
