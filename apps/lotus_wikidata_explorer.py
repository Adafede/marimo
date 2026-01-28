# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     # "epam-indigo>=1.36.1",
#     "great-tables==0.20.0",
#     "marimo",
#     "polars==1.37.1",
#     "rdflib==7.5.0",
# ]
# ///

"""
LOTUS Wikidata Explorer

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
    import re
    import time
    import hashlib
    import sys
    import urllib.request
    import urllib.parse
    from dataclasses import dataclass, field
    from datetime import datetime
    from rdflib import Graph, Literal, URIRef, BNode
    from rdflib.namespace import RDF, RDFS, XSD, DCTERMS
    from typing import Any, Final, TypedDict, cast

    # Toggle this flag for local vs remote development
    _USE_LOCAL = True  # Set to True for local development
    if _USE_LOCAL:
        # Add your local module directory to the path
        # Adjust this path to where your "modules" folder is located locally
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
    )
    from modules.knowledge.wikidata.url.constants import (
        STATEMENT_PREFIX as WIKIDATA_STATEMENT_PREFIX,
    )
    from modules.knowledge.wikidata.url.constants import WIKIDATA_HTTP_BASE
    from modules.knowledge.wikidata.url.constants import WIKI_PREFIX
    from modules.knowledge.wikidata.html.scholia import scholia_url
    from modules.knowledge.wikidata.sparql.query_taxon_search import query_taxon_search
    from modules.knowledge.wikidata.sparql.query_taxon_connectivity import (
        query_taxon_connectivity,
    )
    from modules.knowledge.wikidata.sparql.query_taxon_details import (
        query_taxon_details,
    )
    from modules.knowledge.wikidata.sparql.query_sachem import query_sachem
    from modules.knowledge.wikidata.sparql.query_compounds import (
        query_compounds_by_taxon,
        query_all_compounds,
    )
    from modules.net.sparql.execute_with_retry import execute_with_retry
    from modules.net.sparql.parse_response import parse_sparql_response
    from modules.net.sparql.values_clause import values_clause
    from modules.chem.cdk.depict.svg_from_smiles import svg_from_smiles

    # indigo alternative
    # from modules.chem.indigo.depict.svg_from_mol import svg_from_mol
    # from modules.chem.indigo.mol.mol_from_smiles import mol_from_smiles
    from modules.knowledge.rdf.graph.add_literal import add_literal
    from modules.knowledge.rdf.namespace.wikidata import WIKIDATA_NAMESPACES
    from modules.text.formula.element_config import (
        ELEMENT_DEFAULTS,
    )
    from modules.ui.html_from_image import html_from_image
    from modules.ui.marimo.wrap_html import wrap_html
    from modules.ui.marimo.wrap_image import wrap_image
    from modules.io.compress.if_large import compress_if_large

    # Patch urllib for Pyodide/WASM (browser) compatibility
    IS_PYODIDE = "pyodide" in sys.modules
    if IS_PYODIDE:
        import pyodide_http

        pyodide_http.patch_all()

    # ====================================================================
    # APPLICATION CONFIGURATION
    # ====================================================================

    class ConfigDict(TypedDict):
        app_version: str
        app_name: str
        app_url: str
        qlever_endpoint: str
        wikidata_endpoint: str
        idsm_endpoint: str
        table_row_limit: int
        download_embed_threshold_bytes: int
        color_hyperlink: str
        color_wikidata_blue: str
        color_wikidata_green: str
        color_wikidata_red: str
        page_size_default: int
        page_size_export: int
        year_range_start: int
        year_default_start: int
        mass_default_min: int
        mass_default_max: int
        mass_ui_max: int
        default_search_type: str
        default_similarity_threshold: float
        default_smiles: str
        default_taxon: str

    CONFIG: Final[ConfigDict] = {
        "app_version": "0.1.0",
        "app_name": "LOTUS Wikidata Explorer",
        "app_url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
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
        "page_size_export": 25,
        "year_range_start": 1_700,
        "year_default_start": 1_900,
        "mass_default_min": 0,
        "mass_default_max": 2_000,
        "mass_ui_max": 10_000,
        "default_search_type": "substructure",
        "default_similarity_threshold": 0.8,
        "default_smiles": "",
        "default_taxon": "Gentiana lutea",
    }

    # Pluralization map for irregular forms
    PLURAL_MAP = {
        "Entry": "Entries",
        "entry": "entries",
        "Taxon": "Taxa",
        "taxon": "taxa",
    }


@app.class_definition
@dataclass
class SearchParams:
    """Consolidated search parameters - replaces 30+ individual state variables."""

    # Core search parameters
    taxon: str = "Gentiana lutea"
    smiles: str = ""
    smiles_search_type: str = "substructure"
    smiles_threshold: float = 0.8

    # Mass filter parameters
    mass_filter: bool = False
    mass_min: float = 0.0
    mass_max: float = 2000.0

    # Year filter parameters
    year_filter: bool = False
    year_start: int = 1_900
    year_end: int = field(default_factory=lambda: datetime.now().year)

    # Formula filter parameters
    formula_filter: bool = False
    exact_formula: str = ""

    # Element range parameters (min/max for C, H, N, O, P, S)
    c_min: int | None = None
    c_max: int | None = None
    h_min: int | None = None
    h_max: int | None = None
    n_min: int | None = None
    n_max: int | None = None
    o_min: int | None = None
    o_max: int | None = None
    p_min: int | None = None
    p_max: int | None = None
    s_min: int | None = None
    s_max: int | None = None

    # Halogen filter states
    f_state: str = "allowed"
    cl_state: str = "allowed"
    br_state: str = "allowed"
    i_state: str = "allowed"

    # Auto-run flag (set when URL params are detected)
    auto_run: bool = False

    @classmethod
    def from_url_params(cls, params: dict) -> "SearchParams":
        """Create SearchParams from URL query parameters."""
        if not params or ("taxon" not in params and "smiles" not in params):
            return cls()

        ff = params.get("formula_filter") == "true"
        # Parse element ranges with loop
        elem_vals = {}
        for e in ("c", "h", "n", "o", "p", "s"):
            elem_vals[f"{e}_min"] = (
                int(params[f"{e}_min"]) if f"{e}_min" in params else None
            )
            elem_vals[f"{e}_max"] = (
                int(params[f"{e}_max"]) if f"{e}_max" in params else None
            )

        return cls(
            taxon=params.get("taxon", CONFIG["default_taxon"]),
            smiles=params.get("smiles", CONFIG["default_smiles"]),
            smiles_search_type=params.get(
                "smiles_search_type",
                CONFIG["default_search_type"],
            ),
            smiles_threshold=float(
                params.get("smiles_threshold", CONFIG["default_similarity_threshold"]),
            ),
            mass_filter=params.get("mass_filter") == "true",
            mass_min=float(params.get("mass_min", CONFIG["mass_default_min"])),
            mass_max=float(params.get("mass_max", CONFIG["mass_default_max"])),
            year_filter=params.get("year_filter") == "true",
            year_start=int(params.get("year_start", CONFIG["year_default_start"])),
            year_end=int(params.get("year_end", datetime.now().year)),
            formula_filter=ff,
            exact_formula=params.get("exact_formula", ""),
            **elem_vals,
            f_state=params.get("f_state", "allowed"),
            cl_state=params.get("cl_state", "allowed"),
            br_state=params.get("br_state", "allowed"),
            i_state=params.get("i_state", "allowed"),
            auto_run=True,
        )

    def to_formula_filters(self) -> FormulaFilters | None:
        """Convert to FormulaFilters if formula filter is active."""
        if not self.formula_filter:
            return None
        elem_args = {}
        for e in ("c", "h", "n", "o", "p", "s"):
            min_key = f"{e}_min"
            max_key = f"{e}_max"
            min_v, max_v = getattr(self, min_key), getattr(self, max_key)
            elem_args[min_key] = min_v or 0
            elem_args[max_key] = (
                max_v if max_v is not None else cast(int, CONFIG[f"element_{max_key}"])
            )
        return create_filters(
            exact_formula=self.exact_formula,
            **elem_args,
            f_state=self.f_state,
            cl_state=self.cl_state,
            br_state=self.br_state,
            i_state=self.i_state,
        )


@app.function
def create_taxon_warning_html(
    matches: list,
    selected_qid: str,
    is_exact: bool,
) -> mo.Html:
    """Create an HTML warning with clickable QID links and taxon details."""

    match_type = "exact matches" if is_exact else "similar taxa"
    intro = (
        f"Ambiguous taxon name. Multiple {match_type} found:"
        if is_exact
        else "No exact match. Similar taxa found:"
    )

    # Build HTML list of matches
    items = []
    for match_data in matches:
        qid = match_data[0]
        name = match_data[1]
        description = match_data[2] if len(match_data) > 2 else None
        parent = match_data[3] if len(match_data) > 3 else None
        edges_count = match_data[4] if len(match_data) > 4 else None

        # Create clickable link using module function with custom styling
        link_html = f'<a href="{scholia_url(qid)}" target="_blank" rel="noopener noreferrer" style="color: {CONFIG["color_hyperlink"]}; font-weight: bold;">{qid}</a>'

        # Build details string
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

        # Highlight the selected one
        if qid == selected_qid:
            items.append(
                f"<li>{link_html} {details_str} <strong>< USING THIS ONE (most edges)</strong></li>",
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
        <em>For precision, please use a specific QID directly in the search box. When ambiguous, the taxon with the most compound links is automatically selected.</em>
    </div>
    """

    return mo.Html(html)


@app.function
def resolve_ambiguous_matches(
    matches: list[tuple[str, str]],
    is_exact: bool,
) -> tuple[str, mo.Html]:
    qids = tuple(qid for qid, _ in matches)
    info = {qid: [0, "", "", ""] for qid in qids}

    # --- connectivity ---
    csv_bytes = execute_with_retry(
        query_taxon_connectivity(values_clause("taxon", qids, prefix="wd:")),
        endpoint=CONFIG["qlever_endpoint"],
        fallback_endpoint=None,
    )
    if csv_bytes and csv_bytes.strip():
        for row in parse_sparql_response(csv_bytes).collect().iter_rows(named=True):
            taxon_url = row.get("taxon")
            if taxon_url:
                qid = extract_from_url(taxon_url, WIKIDATA_ENTITY_PREFIX)
                info[qid][0] = int(row.get("count") or 0)
    del csv_bytes

    csv_bytes = execute_with_retry(
        query_taxon_details(values_clause("taxon", qids, prefix="wd:")),
        endpoint=CONFIG["qlever_endpoint"],
        fallback_endpoint=None,
    )
    if csv_bytes and csv_bytes.strip():
        for row in parse_sparql_response(csv_bytes).collect().iter_rows(named=True):
            taxon_url = row.get("taxon")
            if taxon_url:
                qid = extract_from_url(taxon_url, WIKIDATA_ENTITY_PREFIX)
                info[qid][1] = row.get("taxonDescription", "")
                info[qid][2] = row.get("taxon_parentLabel", "")
    del csv_bytes

    # best qid (no sort yet)
    selected_qid = max(qids, key=lambda q: info[q][0])

    # sort once, only for display
    matches_sorted = sorted(
        matches,
        key=lambda x: info[x[0]][0],
        reverse=True,
    )

    matches_with_details = [
        (qid, name, info[qid][1], info[qid][2], info[qid][0])
        for qid, name in matches_sorted
    ]

    return selected_qid, create_taxon_warning_html(
        matches_with_details,
        selected_qid,
        is_exact=is_exact,
    )


@app.function
def resolve_taxon_to_qid(
    taxon_input: str | int,
) -> tuple[str | None, mo.Html | None]:
    """Resolve taxon name or QID (int or string) to a valid QID."""
    taxon_input = taxon_input.strip()

    if taxon_input is None:
        return None, None

    # Convert ints to QID string
    if isinstance(taxon_input, int):
        return f"Q{taxon_input}", None

    # Ensure we have a stripped string
    taxon_input = str(taxon_input).strip()

    # Handle wildcard for all taxa
    if taxon_input == "*":
        return "*", None

    # Early return if input is already a QID like "Q42"
    if taxon_input.upper().startswith("Q") and taxon_input[1:].isdigit():
        return taxon_input.upper(), None

    # Search for taxon by name (CSV for memory efficiency)
    try:
        query = query_taxon_search(taxon_input)
        csv_bytes = execute_with_retry(
            query,
            endpoint=CONFIG["qlever_endpoint"],
            fallback_endpoint=None,
        )

        if not csv_bytes or not csv_bytes.strip():
            return None, None

        df = parse_sparql_response(csv_bytes).collect()
        del csv_bytes

        if df.is_empty():
            return None, None

        # Extract matches
        matches = [
            (
                extract_from_url(row["taxon"], WIKIDATA_ENTITY_PREFIX),
                row["taxon_name"],
            )
            for row in df.iter_rows(named=True)
            if row.get("taxon") and row.get("taxon_name")
        ]

        del df

        if not matches:
            return None, None

        # Find exact matches (case-insensitive)
        taxon_lower = taxon_input.lower()
        exact_matches = [
            (qid, name) for qid, name in matches if name.lower() == taxon_lower
        ]

        # Single exact match - perfect
        if len(exact_matches) == 1:
            return exact_matches[0][0], None

        # Multiple exact matches - need to disambiguate
        if len(exact_matches) > 1:
            return resolve_ambiguous_matches(exact_matches, is_exact=True)

        # No exact match but multiple similar matches - use best one with warning
        if len(matches) > 1:
            return resolve_ambiguous_matches(matches, is_exact=False)

        return matches[0][0], None

    except Exception:
        return None, None


@app.function
def build_active_filters_dict(
    mass_filter_active: bool,
    mass_min_val: float | None,
    mass_max_val: float | None,
    year_filter_active: bool,
    year_start_val: int | None,
    year_end_val: int | None,
    formula_filters: FormulaFilters | None,
    smiles: str | None = None,
    smiles_search_type: str | None = None,
    smiles_threshold: float | None = None,
) -> dict[str, Any]:
    """Build a dictionary of active filters for metadata export."""
    filters = {}

    # Chemical structure (SMILES) search
    if smiles and smiles.strip():
        filters["chemical_structure"] = {
            "smiles": smiles.strip(),
            "search_type": smiles_search_type or "substructure",
        }
        # Add threshold only for similarity searches
        if smiles_search_type == "similarity" and smiles_threshold is not None:
            filters["chemical_structure"]["similarity_threshold"] = smiles_threshold

    # Mass filter
    if mass_filter_active and (mass_min_val is not None or mass_max_val is not None):
        filters["mass"] = {
            "min": mass_min_val,
            "max": mass_max_val,
        }

    # Year filter
    if year_filter_active and (year_start_val is not None or year_end_val is not None):
        filters["publication_year"] = {
            "start": year_start_val,
            "end": year_end_val,
        }

    # Formula filter
    formula_dict = serialize_filters(formula_filters)
    if formula_dict:
        filters["molecular_formula"] = formula_dict

    return filters


@app.function
def generate_filename(
    taxon_name: str,
    file_type: str,
    prefix: str = "lotus_data",
    filters: dict[str, Any] | None = None,
) -> str:
    """Generate standardized, descriptive filename for exports."""
    # Handle wildcard for all taxa
    if taxon_name == "*":
        safe_name = "all_taxa"
    else:
        # Replace spaces and special characters
        safe_name = taxon_name.replace(" ", "_").replace("/", "_")

    # Build filename components
    components = [prefix, safe_name]

    # Add SMILES search type if present
    if filters and "chemical_structure" in filters:
        search_type = filters["chemical_structure"].get(
            "search_type",
            "substructure",
        )
        components.append(search_type)  # Just the type, not "smiles_type"

    # Add general filter indicator if other filters are active
    other_filters = {
        k: v for k, v in (filters or {}).items() if k != "chemical_structure"
    }
    if other_filters:
        components.append("filtered")

    date_str = datetime.now().strftime("%Y%m%d")
    filename_base = "_".join(components)
    return f"{date_str}_{filename_base}.{file_type}"


@app.function
def build_api_url(
    taxon: str,
    smiles: str,
    smiles_search_type: str,
    smiles_threshold: float,
    mass_filter: bool,
    mass_min: float,
    mass_max: float,
    year_filter: bool,
    year_start: int,
    year_end: int,
    formula_filter: bool,
    exact_formula: str,
    c_min: int,
    c_max: int,
    h_min: int,
    h_max: int,
    n_min: int,
    n_max: int,
    o_min: int,
    o_max: int,
    p_min: int,
    p_max: int,
    s_min: int,
    s_max: int,
    f_state: str,
    cl_state: str,
    br_state: str,
    i_state: str,
) -> str:
    """Build a shareable API URL from current search parameters."""
    params = {}

    # Taxon parameter
    if taxon and taxon.strip():
        params["taxon"] = taxon.strip()

    # SMILES parameters
    if smiles and smiles.strip():
        params["smiles"] = smiles.strip()
        params["smiles_search_type"] = smiles_search_type
        if smiles_search_type == "similarity":
            params["smiles_threshold"] = str(smiles_threshold)

    # Mass filter
    if mass_filter:
        params["mass_filter"] = "true"
        if mass_min is not None:
            params["mass_min"] = str(mass_min)
        if mass_max is not None:
            params["mass_max"] = str(mass_max)

    # Year filter
    if year_filter:
        params["year_filter"] = "true"
        if year_start is not None:
            params["year_start"] = str(year_start)
        if year_end is not None:
            params["year_end"] = str(year_end)

    # Formula filter
    if formula_filter:
        params["formula_filter"] = "true"
        if exact_formula and exact_formula.strip():
            params["exact_formula"] = exact_formula.strip()

        # Element ranges (only add non-default values)
        for elem in ["c", "h", "n", "o", "p", "s"]:
            min_val = locals()[f"{elem}_min"]
            max_val = locals()[f"{elem}_max"]
            if min_val > 0:
                params[f"{elem}_min"] = str(min_val)
            if max_val != ELEMENT_DEFAULTS[elem]:
                params[f"{elem}_max"] = str(max_val)

        # Halogen states (only add non-default)
        for halogen, state in [
            ("f", f_state),
            ("cl", cl_state),
            ("br", br_state),
            ("i", i_state),
        ]:
            if state != "allowed":
                params[f"{halogen}_state"] = state

    # Build URL
    if params:
        query_string = urllib.parse.urlencode(params)
        return f"?{query_string}"
    else:
        return ""


@app.function
def create_download_button(data: str, filename: str, label: str, base_mimetype: str):
    """Create a download button with automatic compression for large files."""
    # Convert to bytes if needed
    data_bytes = data.encode("utf-8") if isinstance(data, str) else data
    compressed_data, was_compressed = compress_if_large(
        data_bytes,
        CONFIG["download_embed_threshold_bytes"],
    )

    # Add compression indicator to label
    display_label = label + (" (gzipped)" if was_compressed else "")

    # Update filename extension if compressed
    final_filename = filename + ".gz" if was_compressed else filename
    final_mimetype = "application/gzip" if was_compressed else base_mimetype

    # Use standard mo.download - works on desktop and most mobile
    return mo.download(
        data=compressed_data,
        filename=final_filename,
        label=display_label,
        mimetype=final_mimetype,
    )


@app.function
def query_wikidata(
    qid: str,
    year_start: int | None = None,
    year_end: int | None = None,
    mass_min: float | None = None,
    mass_max: float | None = None,
    formula_filters: FormulaFilters | None = None,
    smiles: str | None = None,
    smiles_search_type: str = "substructure",
    smiles_threshold: float = 0.8,
    endpoint: str = CONFIG["qlever_endpoint"],
) -> pl.LazyFrame:
    """
    Query Wikidata and return LazyFrame (NOT materialized).
    """
    # Validation
    if smiles_search_type not in ("substructure", "similarity"):
        raise ValueError(f"Invalid smiles_search_type: '{smiles_search_type}'")

    if not (0.0 <= smiles_threshold <= 1.0):
        raise ValueError(f"Invalid smiles_threshold: {smiles_threshold}")

    # Build query
    if smiles:
        query = query_sachem(
            escaped_smiles=validate_and_escape(smiles),
            search_type=smiles_search_type,
            threshold=smiles_threshold,
            taxon_qid=qid if qid != "*" else None,
        )
    elif qid == "*":
        query = query_all_compounds()
    else:
        query = query_compounds_by_taxon(qid)

    # Execute query
    csv_bytes = execute_with_retry(query, endpoint)

    if not csv_bytes or csv_bytes.strip() == b"":
        return pl.LazyFrame()

    # Parse to LazyFrame (NO materialization)
    lazy_df = pl.scan_csv(
        io.BytesIO(csv_bytes),
        low_memory=True,
        rechunk=False,
    )

    # Rename
    lazy_df = lazy_df.rename(
        {
            "compoundLabel": "name",
            "compound_inchikey": "inchikey",
            "ref_qid": "reference",
            "ref_date": "pub_date",
            "compound_mass": "mass",
            "compound_formula": "mf",
        },
    )

    # Combine SMILES
    lazy_df = lazy_df.with_columns(
        [
            pl.coalesce(["compound_smiles_iso", "compound_smiles_conn"]).alias(
                "smiles",
            ),
        ],
    )

    # DOI extraction
    lazy_df = lazy_df.with_columns(
        [
            pl.when(pl.col("ref_doi").str.starts_with("http"))
            .then(pl.col("ref_doi").str.split("doi.org/").list.last())
            .otherwise(pl.col("ref_doi"))
            .alias("ref_doi"),
        ],
    )

    # Date parsing
    lazy_df = lazy_df.with_columns(
        [
            pl.when(pl.col("pub_date").is_not_null() & (pl.col("pub_date") != ""))
            .then(
                pl.col("pub_date")
                .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
                .dt.date(),
            )
            .otherwise(None)
            .alias("pub_date"),
        ],
    )

    # Mass to Float32
    lazy_df = lazy_df.with_columns(
        [
            pl.col("mass").cast(pl.Float32, strict=False),
        ],
    )

    # Drop old columns
    columns_to_drop = ["compound_smiles_iso", "compound_smiles_conn"]
    lazy_df = lazy_df.drop(
        [col for col in columns_to_drop if col in lazy_df.collect_schema().names()],
    )

    # Year filter
    if year_start:
        lazy_df = lazy_df.filter(pl.col("pub_date").dt.year() >= year_start)
    if year_end:
        lazy_df = lazy_df.filter(pl.col("pub_date").dt.year() <= year_end)

    # Mass filter
    if mass_min:
        lazy_df = lazy_df.filter(pl.col("mass") >= mass_min)
    if mass_max:
        lazy_df = lazy_df.filter(pl.col("mass") <= mass_max)

    # Formula filter
    if (
        formula_filters
        and hasattr(formula_filters, "is_active")
        and formula_filters.is_active()
    ):
        lazy_df = lazy_df.filter(
            pl.col("mf").map_batches(
                lambda s: s.map_elements(
                    lambda f: match_filters(f or "", formula_filters),
                    return_dtype=pl.Boolean,
                ),
            ),
        )

    # Add missing columns (lazy)
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
    missing = [col for col in required if col not in lazy_df.collect_schema().names()]
    if missing:
        lazy_df = lazy_df.with_columns([pl.lit(None).alias(col) for col in missing])

    # Deduplicate and sort
    lazy_df = lazy_df.unique(
        subset=["compound", "taxon", "reference"],
        keep="first",
    ).sort("name")

    # Return LazyFrame - NO .collect()!
    return lazy_df


@app.function
def build_display_dataframe(df: pl.LazyFrame) -> pl.DataFrame:
    """Build display DataFrame with HTML-formatted columns."""

    # Pre-compute colors (avoid repeated dictionary lookups)
    color_compound = CONFIG["color_wikidata_red"]
    color_taxon = CONFIG["color_wikidata_green"]
    color_ref = CONFIG["color_wikidata_blue"]
    color_link = CONFIG["color_hyperlink"]
    limit = CONFIG["table_row_limit"]

    # Physical limit
    if limit:
        df = df.limit(limit)

    # Generate molecule images (small Python loop)
    df = df.with_columns(
        pl.col("smiles")
        .map_elements(
            lambda s: mo.image(svg_from_smiles(s)), return_dtype=pl.String if s else ""
        )
        .alias("Compound Depiction")
    )

    # Add vectorized HTML columns in small groups
    df = df.with_columns(
        [
            # First batch
            pl.when(pl.col("compound").is_not_null())
            .then(
                pl.format(
                    '<a href="https://scholia.toolforge.org/Q{}" style="color:{};">Q{}</a>',
                    pl.col("compound"),
                    pl.lit(CONFIG["color_wikidata_red"]),
                    pl.col("compound"),
                )
            )
            .otherwise(pl.lit(""))
            .alias("Compound QID"),
            pl.when(pl.col("taxon").is_not_null())
            .then(
                pl.format(
                    '<a href="https://scholia.toolforge.org/Q{}" style="color:{};">Q{}</a>',
                    pl.col("taxon"),
                    pl.lit(CONFIG["color_wikidata_green"]),
                    pl.col("taxon"),
                )
            )
            .otherwise(pl.lit(""))
            .alias("Taxon QID"),
        ]
    )

    # Add remaining columns in a second batch
    df = df.with_columns(
        [
            pl.when(pl.col("reference").is_not_null())
            .then(
                pl.format(
                    '<a href="https://scholia.toolforge.org/Q{}" style="color:{};">Q{}</a>',
                    pl.col("reference"),
                    pl.lit(CONFIG["color_wikidata_blue"]),
                    pl.col("reference"),
                )
            )
            .otherwise(pl.lit(""))
            .alias("Reference QID"),
            # DOI links
            pl.when(pl.col("ref_doi").is_not_null() & (pl.col("ref_doi") != ""))
            .then(
                pl.format(
                    '<a href="https://doi.org/{}" style="color:{};">{}</a>',
                    pl.col("ref_doi"),
                    pl.lit(CONFIG["color_hyperlink"]),
                    pl.col("ref_doi"),
                )
            )
            .otherwise(pl.lit(""))
            .alias("Reference DOI"),
        ]
    )

    return df


@app.function
def prepare_export_dataframe(
    lazy_df: pl.LazyFrame,
    include_rdf_ref: bool = False,
) -> pl.LazyFrame:
    """
    Prepare export transformations lazily.

    Returns LazyFrame ready for .write_csv()/.write_json().
    Those methods call .collect() internally.
    """
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
        # QIDs (lazy concat)
        pl.concat_str([pl.lit("Q"), pl.col("compound").cast(pl.Utf8)]).alias(
            "compound_qid",
        ),
        pl.concat_str([pl.lit("Q"), pl.col("taxon").cast(pl.Utf8)]).alias("taxon_qid"),
        pl.concat_str([pl.lit("Q"), pl.col("reference").cast(pl.Utf8)]).alias(
            "reference_qid",
        ),
    ]

    # Statement
    if "statement" in lazy_df.collect_schema().names():
        exprs.append(
            pl.col("statement")
            .str.replace(WIKIDATA_STATEMENT_PREFIX, "", literal=True)
            .alias("statement_id"),
        )

    # RDF ref
    if include_rdf_ref and "ref" in lazy_df.collect_schema().names():
        exprs.append(pl.col("ref"))

    return lazy_df.select(exprs)


@app.function
def create_export_metadata(
    df: pl.DataFrame,
    counts: dict,
    taxon_input: str,
    qid: str,
    filters: dict[str, Any],
    query_hash: str | None = None,
    result_hash: str | None = None,
) -> dict[str, Any]:
    """Create FAIR-compliant metadata for exported datasets."""

    # Build descriptive name and description based on search type
    smiles_info = filters.get("chemical_structure", {}) if filters else {}

    if smiles_info:
        search_type = smiles_info.get("search_type", "substructure")
        smiles_str = smiles_info.get("smiles", "")

        if qid:
            # Combined search
            dataset_name = f"LOTUS Data - {search_type.title()} search in {taxon_input}"
            description = (
                f"Chemical compounds matching {search_type} search "
                f"(SMILES: {smiles_str}) "
                f"within taxon {taxon_input} (Wikidata QID: {qid}). "
            )
        else:
            # SMILES-only search
            dataset_name = f"LOTUS Data - Chemical {search_type.title()} Search"
            description = (
                f"Chemical compounds matching {search_type} search "
                f"(SMILES: {smiles_str}). "
            )

        if search_type == "similarity":
            threshold = smiles_info.get("similarity_threshold", 0.8)
            description += f"Tanimoto similarity threshold: {threshold}. "
    else:
        # Taxon-only search
        dataset_name = f"LOTUS Data - {taxon_input}"
        description = f"Chemical compounds from taxon {taxon_input} " + (
            f"(Wikidata QID: {qid}). " if qid else ". "
        )

    description += "Retrieved via LOTUS Wikidata Explorer with chemical search capabilities (SACHEM/IDSM)."

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
            {"@type": "Organization", "name": n, "url": u}
            for n, u in [
                ("LOTUS Initiative", WIKI_PREFIX + "Q104225190"),
                ("Wikidata", WIKIDATA_HTTP_BASE),
                ("IDSM", "https://idsm.elixir-czech.cz/"),
            ]
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
                "encodingFormat": f,
                "contentUrl": f"data:{f}",
            }
            for f in ["text/csv", "application/json", "text/turtle"]
        ],
        "numberOfRecords": counts["n_entries"],
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
        "search_parameters": {"taxon": taxon_input, "taxon_qid": qid},
        "chemical_search_service": {
            "name": "SACHEM",
            "provider": "IDSM",
            "endpoint": CONFIG["idsm_endpoint"],
        },
    }

    if filters:
        metadata["search_parameters"]["filters"] = filters
    if query_hash or result_hash:
        metadata["provenance"] = {}
        if query_hash:
            metadata["provenance"]["query_hash"] = {
                "algorithm": "SHA-256",
                "value": query_hash,
            }
        if result_hash:
            metadata["provenance"]["result_hash"] = {
                "algorithm": "SHA-256",
                "value": result_hash,
            }
            metadata["provenance"]["dataset_uri"] = f"urn:hash:sha256:{result_hash}"

    return metadata


@app.function
def create_citation_text(taxon_input: str) -> str:
    """Generate citation text for the exported data."""
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

### This Tool
{CONFIG["app_name"]} v{CONFIG["app_version"]}
[Source Code]({CONFIG["app_url"]}) (AGPL-3.0)

### Data Sources
- **LOTUS Initiative**: [Q104225190](https://www.wikidata.org/wiki/Q104225190) - CC0 1.0
- **Wikidata**: [www.wikidata.org](https://www.wikidata.org/) - CC0 1.0
"""


@app.function
def compute_provenance_hashes(
    qid: str | None,
    taxon_input: str | None,
    filters: dict[str, Any] | None,
    df: pl.LazyFrame,
) -> tuple[str, str]:
    """
    Compute query and result hashes for provenance tracking.

    Returns:
        Tuple of (query_hash, result_hash) where:
        - query_hash: based on search parameters (what was asked)
        - result_hash: based on compound identifiers (what was found)
    """
    # Query hash - small data, use direct approach
    query_components = [qid or "", taxon_input or ""]
    if filters:
        query_components.append(json.dumps(filters, sort_keys=True))
    query_hash = hashlib.sha256(
        "|".join(query_components).encode("utf-8"),
    ).hexdigest()

    # Result hash - streaming approach for memory efficiency
    result_hasher = hashlib.sha256()
    compound_col = (
        "compound_qid" if "compound_qid" in df.collect_schema().names() else "compound"
    )

    if compound_col in df.collect_schema().names():
        try:
            # Get unique compound IDs as a sorted series (no intermediate list)
            unique_ids = (
                df.select(
                    pl.col(compound_col)
                    .cast(pl.Utf8)
                    .str.replace(WIKIDATA_ENTITY_PREFIX, "", literal=True),
                )
                .to_series()
                .drop_nulls()
                .unique()
                .sort()
                .collect()
            )
            # Stream through values without creating full string
            for i, val in enumerate(unique_ids):
                if i > 0:
                    result_hasher.update(b"|")
                if val:
                    result_hasher.update(val.encode("utf-8"))
        except Exception:
            pass

    return query_hash, result_hasher.hexdigest()


@app.function
def create_dataset_uri(
    qid: str,
    taxon_input: str,
    filters: dict[str, Any] | None,
    df: pl.DataFrame,
) -> tuple[URIRef, str, str]:
    """
    Create dataset URI based on result content for reproducibility.

    NOTE: LOTUS data is hosted on Wikidata (https://www.wikidata.org/wiki/Q104225190).
    There is no separate LOTUS namespace - data is stored as regular Wikidata entities.
    This export creates a virtual dataset URI for the query result using a
    content-addressable URN based ONLY on what was found (not what was asked).
    The query hash is returned separately for metadata storage.
    """
    query_hash, result_hash = compute_provenance_hashes(
        qid,
        taxon_input,
        filters,
        df,
    )

    # Create a content-addressable URI using URN scheme with ONLY result hash
    # Format: urn:hash:sha256:RESULT_HASH
    # This identifies the dataset by its content, not by how it was obtained
    dataset_uri = URIRef(f"urn:hash:sha256:{result_hash}")

    return dataset_uri, query_hash, result_hash


@app.function
def build_dataset_description(
    taxon_input: str,
    filters: dict[str, Any],
) -> tuple[str, str]:
    """Build descriptive dataset name and description. Returns (name, description)."""
    dataset_name = f"LOTUS Data for {taxon_input}"
    dataset_desc = f"Chemical compounds found in taxon {taxon_input} from Wikidata"

    if filters:
        smiles_info = filters.get("chemical_structure", {})
        if smiles_info:
            search_type = smiles_info.get("search_type", "substructure")
            dataset_desc += f" using {search_type} search"

        if filters.get("mass"):
            mass = filters["mass"]
            dataset_desc += (
                f", mass filter: {mass.get('min', 'N/A')}-{mass.get('max', 'N/A')} Da"
            )

        if filters.get("molecular_formula"):
            dataset_desc += ", molecular formula constraints applied"

    return dataset_name, dataset_desc


@app.function
def add_dataset_metadata(
    g: Graph,
    dataset_uri: URIRef,
    dataset_name: str,
    dataset_desc: str,
    qid: str,
    df_len: int,
    query_hash: str,
    result_hash: str,
) -> None:
    """Add core dataset metadata to RDF graph (mutates graph in-place)."""
    SCHEMA = WIKIDATA_NAMESPACES["SCHEMA"]
    WD = WIKIDATA_NAMESPACES["WD"]

    # Dataset type and basic metadata
    g.add((dataset_uri, RDF.type, SCHEMA.Dataset))
    g.add((dataset_uri, SCHEMA.name, Literal(dataset_name, datatype=XSD.string)))
    g.add(
        (
            dataset_uri,
            SCHEMA.description,
            Literal(dataset_desc, datatype=XSD.string),
        ),
    )

    # License and provenance - CC0 from Wikidata/LOTUS
    g.add(
        (
            dataset_uri,
            SCHEMA.license,
            URIRef("https://creativecommons.org/publicdomain/zero/1.0/"),
        ),
    )
    g.add((dataset_uri, SCHEMA.provider, URIRef(CONFIG["app_url"])))
    g.add((dataset_uri, DCTERMS.source, URIRef(WIKIDATA_HTTP_BASE)))

    # Dataset statistics and versioning
    g.add(
        (
            dataset_uri,
            SCHEMA.numberOfRecords,
            Literal(df_len, datatype=XSD.integer),
        ),
    )
    g.add(
        (
            dataset_uri,
            SCHEMA.version,
            Literal(CONFIG["app_version"], datatype=XSD.string),
        ),
    )

    # Reference to LOTUS Initiative (Q104225190) as the source project
    g.add(
        (
            dataset_uri,
            SCHEMA.isBasedOn,
            URIRef(WIKI_PREFIX + "Q104225190"),
        ),
    )

    # Link to the taxon being queried (if specific)
    if qid and qid != "*":
        g.add((dataset_uri, SCHEMA.about, WD[qid]))

    # Add provenance hashes for reproducibility
    # Query hash stored as additional metadata (how dataset was generated)
    g.add(
        (
            dataset_uri,
            DCTERMS.provenance,
            Literal(
                f"Generated by query with hash: {query_hash}",
                datatype=XSD.string,
            ),
        ),
    )
    # Result hash is implicit in the dataset URI itself (urn:hash:sha256:RESULT_HASH)
    # but we also add it explicitly for clarity
    g.add(
        (
            dataset_uri,
            DCTERMS.identifier,
            Literal(f"sha256:{result_hash}", datatype=XSD.string),
        ),
    )


@app.function
def add_compound_triples(
    g: Graph,
    row: dict[str, Any],
    dataset_uri: URIRef,
    processed_taxa: set,
    processed_refs: set,
    ns_cache: dict,
) -> None:
    """Add all triples for a single compound using Wikidata's full RDF structure."""
    WD = ns_cache["WD"]
    WDT = ns_cache["WDT"]
    P = ns_cache["P"]
    PS = ns_cache["PS"]
    PR = ns_cache["PR"]
    PROV = ns_cache["PROV"]
    SCHEMA = ns_cache["SCHEMA"]

    compound_qid = row.get("compound_qid", "")
    if not compound_qid:
        return

    compound_uri = WD[compound_qid]

    # Link compound to dataset
    g.add((dataset_uri, SCHEMA.hasPart, compound_uri))

    # Compound identifiers using Wikidata properties (direct properties)
    add_literal(
        g,
        compound_uri,
        WDT.P235,
        row.get("compound_inchikey"),
    )  # InChIKey
    add_literal(
        g,
        compound_uri,
        WDT.P233,
        row.get("compound_smiles"),
    )  # Canonical SMILES
    add_literal(
        g,
        compound_uri,
        WDT.P274,
        row.get("molecular_formula"),
    )  # Molecular formula

    # Mass (P2067)
    if row.get("compound_mass") is not None:
        add_literal(
            g,
            compound_uri,
            WDT.P2067,
            row["compound_mass"],
            XSD.float,
        )

    # Compound label
    add_literal(g, compound_uri, RDFS.label, row.get("compound_name"))

    # Taxonomic association using P703 (found in taxon) with FULL STATEMENT STRUCTURE
    # This mirrors the SPARQL query pattern:
    #   ?compound p:P703 ?statement.
    #   ?statement ps:P703 ?taxon;
    #              prov:wasDerivedFrom ?ref.
    #   ?ref pr:P248 ?ref_qid.
    taxon_qid = row.get("taxon_qid")
    ref_qid = row.get("reference_qid")
    statement_uri = row.get("statement")
    ref_uri_str = row.get("ref")

    if taxon_qid:
        taxon_uri = WD[taxon_qid]

        # Use actual statement URI from Wikidata if available, otherwise use blank node
        if statement_uri and statement_uri.strip():
            statement_node = URIRef(statement_uri)
        else:
            statement_node = BNode()

        # Full statement pattern (following Wikidata RDF structure)
        g.add(
            (compound_uri, P.P703, statement_node),
        )  # compound has a P703 statement
        g.add((statement_node, PS.P703, taxon_uri))  # statement value is the taxon

        # Add provenance if reference exists
        if ref_qid:
            ref_uri = WD[ref_qid]

            # Use actual reference URI from Wikidata if available, otherwise use blank node
            if ref_uri_str and ref_uri_str.strip():
                ref_node = URIRef(ref_uri_str)
            else:
                ref_node = BNode()

            # Link statement to reference via provenance
            g.add((statement_node, PROV.wasDerivedFrom, ref_node))

            # Reference stated in (pr:P248)
            g.add((ref_node, PR.P248, ref_uri))

            # Add reference metadata once per unique reference
            if ref_qid not in processed_refs:
                # P1476: title
                add_literal(
                    g,
                    ref_uri,
                    WDT.P1476,
                    row.get("reference_title"),
                )
                add_literal(
                    g,
                    ref_uri,
                    RDFS.label,
                    row.get("reference_title"),
                )

                # P356: DOI
                if row.get("reference_doi"):
                    add_literal(
                        g,
                        ref_uri,
                        WDT.P356,
                        row.get("reference_doi"),
                    )

                # P577: publication date
                if row.get("reference_date"):
                    add_literal(
                        g,
                        ref_uri,
                        WDT.P577,
                        str(row["reference_date"]),
                        XSD.date,
                    )

                processed_refs.add(ref_qid)

        # Also add the simplified direct triple for convenience (wdt: namespace)
        g.add((compound_uri, WDT.P703, taxon_uri))

        # Add taxon metadata once per unique taxon
        if taxon_qid not in processed_taxa:
            # P225: taxon name
            add_literal(g, taxon_uri, WDT.P225, row.get("taxon_name"))
            add_literal(g, taxon_uri, RDFS.label, row.get("taxon_name"))
            processed_taxa.add(taxon_qid)


@app.function
def export_to_rdf_turtle(
    df: pl.DataFrame | pl.LazyFrame,
    taxon_input: str,
    qid: str,
    filters: dict[str, Any] | None = None,
) -> str:
    """Export data to RDF Turtle format using Wikidata's full RDF structure."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    # Initialize graph
    g = Graph()

    # Bind namespaces from WIKIDATA_NAMESPACES
    g.bind("wd", WIKIDATA_NAMESPACES["WD"])
    g.bind("wdref", WIKIDATA_NAMESPACES["WDREF"])
    g.bind("wds", WIKIDATA_NAMESPACES["WDS"])
    g.bind("wdt", WIKIDATA_NAMESPACES["WDT"])
    g.bind("p", WIKIDATA_NAMESPACES["P"])
    g.bind("ps", WIKIDATA_NAMESPACES["PS"])
    g.bind("pr", WIKIDATA_NAMESPACES["PR"])
    g.bind("prov", WIKIDATA_NAMESPACES["PROV"])
    g.bind("schema", WIKIDATA_NAMESPACES["SCHEMA"])
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("dcterms", DCTERMS)

    # Pre-cache namespace references (avoid repeated dict lookups per row)
    ns_cache = {
        "WD": WIKIDATA_NAMESPACES["WD"],
        "WDT": WIKIDATA_NAMESPACES["WDT"],
        "P": WIKIDATA_NAMESPACES["P"],
        "PS": WIKIDATA_NAMESPACES["PS"],
        "PR": WIKIDATA_NAMESPACES["PR"],
        "PROV": WIKIDATA_NAMESPACES["PROV"],
        "SCHEMA": WIKIDATA_NAMESPACES["SCHEMA"],
    }

    # Create dataset URI with provenance hashes
    dataset_uri, query_hash, result_hash = create_dataset_uri(
        qid,
        taxon_input,
        filters,
        df,
    )

    # Build dataset description
    dataset_name, dataset_desc = build_dataset_description(taxon_input, filters)

    # Add dataset metadata
    add_dataset_metadata(
        g,
        dataset_uri,
        dataset_name,
        dataset_desc,
        qid,
        len(df),
        query_hash,
        result_hash,
    )

    # Track unique entities to avoid redundant triples (efficiency)
    processed_taxa = set()
    processed_refs = set()

    # Add compound data with cached namespaces
    for row in df.iter_rows(named=True):
        add_compound_triples(
            g,
            row,
            dataset_uri,
            processed_taxa,
            processed_refs,
            ns_cache,
        )

    # Serialize to Turtle format
    return g.serialize(format="turtle")


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
def ui_url_api():
    # Compact help - collapsed by default
    help_section = mo.accordion(
        {
            "Help & API": mo.md(
                """
                **Search:** Enter a taxon name (e.g., *Gentiana lutea*) and/or a SMILES structure, then click Search.

                **URL API:** `?taxon=Salix&smiles=CC(=O)Oc1ccccc1C(=O)O` | `?taxon=*&mass_filter=true&mass_min=300`
                """,
            ),
        },
    )
    help_section
    return


@app.cell
def url_params_check():
    # URL parameter detection - simple notification
    _url_params_check = mo.query_params()
    if _url_params_check and (
        "taxon" in _url_params_check or "smiles" in _url_params_check
    ):
        taxon = _url_params_check.get("taxon", "")
        smiles = _url_params_check.get("smiles", "")
        msg = f"**Auto-search:** {taxon}" if taxon else ""
        if smiles:
            msg += f" SMILES: `{smiles}`"
        mo.callout(mo.md(msg), kind="info").style(
            style={
                "overflow-wrap": "anywhere",
            },
        )
    return


@app.cell
def ui_search_params(search_params):
    ## TAXON INPUT
    taxon_input = mo.ui.text(
        value=search_params.taxon,
        label="Taxon Name or Wikidata QID - Optional",
        placeholder="e.g., Gentiana lutea, Q157115, or * for all",
        full_width=True,
    )

    ## SMILES INPUT
    smiles_input = mo.ui.text(
        value=search_params.smiles,
        label="Chemical Structure (SMILES) - Optional",
        placeholder="e.g., c1ccccc1 or CC(=O)Oc1ccccc1C(=O)O",
        full_width=True,
    )

    smiles_search_type = mo.ui.dropdown(
        options=["substructure", "similarity"],
        value=search_params.smiles_search_type,
        label="Search Type",
        full_width=True,
    )

    smiles_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=search_params.smiles_threshold,
        label="Similarity Threshold",
        full_width=True,
    )

    ## MASS FILTERS
    mass_filter = mo.ui.checkbox(
        label="Mass filter",
        value=search_params.mass_filter,
    )

    mass_min = mo.ui.number(
        value=search_params.mass_min,
        start=0,
        stop=CONFIG["mass_ui_max"],
        step=0.001,
        label="Min (Da)",
        full_width=True,
    )
    mass_max = mo.ui.number(
        value=search_params.mass_max,
        start=0,
        stop=CONFIG["mass_ui_max"],
        step=0.001,
        label="Max (Da)",
        full_width=True,
    )

    formula_filter = mo.ui.checkbox(
        label="Formula filter",
        value=search_params.formula_filter,
    )
    exact_formula = mo.ui.text(
        value=search_params.exact_formula,
        label="Formula",
        placeholder="C15H10O5",
        full_width=True,
    )

    # Element min/max inputs
    def _mk(element: str, min_val, max_val):
        default_max = ELEMENT_DEFAULTS[element.lower()]
        return (
            mo.ui.number(
                value=min_val,
                start=0,
                stop=default_max,
                label=f"{element} min",
                full_width=True,
            ),
            mo.ui.number(
                value=max_val if max_val is not None else default_max,
                start=0,
                stop=default_max,
                label=f"{element} max",
                full_width=True,
            ),
        )

    c_min, c_max = _mk("C", search_params.c_min, search_params.c_max)
    h_min, h_max = _mk("H", search_params.h_min, search_params.h_max)
    n_min, n_max = _mk("N", search_params.n_min, search_params.n_max)
    o_min, o_max = _mk("O", search_params.o_min, search_params.o_max)
    p_min, p_max = _mk("P", search_params.p_min, search_params.p_max)
    s_min, s_max = _mk("S", search_params.s_min, search_params.s_max)

    # Halogen selectors
    _ho = ["allowed", "required", "excluded"]
    f_state = mo.ui.dropdown(
        options=_ho,
        value=search_params.f_state,
        label="F",
        full_width=True,
    )
    cl_state = mo.ui.dropdown(
        options=_ho,
        value=search_params.cl_state,
        label="Cl",
        full_width=True,
    )
    br_state = mo.ui.dropdown(
        options=_ho,
        value=search_params.br_state,
        label="Br",
        full_width=True,
    )
    i_state = mo.ui.dropdown(
        options=_ho,
        value=search_params.i_state,
        label="I",
        full_width=True,
    )

    current_year = datetime.now().year
    year_filter = mo.ui.checkbox(
        label="Year filter",
        value=search_params.year_filter,
    )
    year_start = mo.ui.number(
        value=search_params.year_start,
        start=CONFIG["year_range_start"],
        stop=current_year,
        label="From",
        full_width=True,
    )
    year_end = mo.ui.number(
        value=search_params.year_end,
        start=CONFIG["year_range_start"],
        stop=current_year,
        label="To",
        full_width=True,
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
def ui_filters(
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
    # Build structure search section
    structure_fields = [smiles_input, smiles_search_type]
    if smiles_search_type.value == "similarity":
        structure_fields.append(smiles_threshold)

    # Main search: search + taxon + SMILES side by side
    main_search = mo.hstack(
        [mo.vstack([run_button, taxon_input]), mo.vstack(structure_fields)],
        gap=2,
        widths="equal",
    )

    # Filter checkboxes inline
    filter_row = mo.hstack(
        [mass_filter, year_filter, formula_filter],
        gap=2,
        justify="start",
    )

    # Build filters UI - compact
    filters_ui = [main_search, filter_row]

    # Conditional filter fields - all inline
    if mass_filter.value:
        filters_ui.append(mo.hstack([mass_min, mass_max], gap=2, widths="equal"))

    if year_filter.value:
        filters_ui.append(mo.hstack([year_start, year_end], gap=2, widths="equal"))

    if formula_filter.value:
        filters_ui.extend(
            [
                exact_formula,
                mo.hstack([c_min, c_max, h_min, h_max, n_min, n_max], gap=1),
                mo.hstack([o_min, o_max, p_min, p_max, s_min, s_max], gap=1),
                mo.hstack([f_state, cl_state, br_state, i_state], gap=1),
            ],
        )

    mo.vstack(filters_ui, gap=1)
    return


@app.cell
def launch_query(
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
    search_params,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    """Execute query and return LazyFrame (NOT materialized)."""
    # Auto-run if URL parameters or button clicked
    if not run_button.value and not search_params.auto_run:
        lazy_results = None
        qid = None
        taxon_warning = None
    else:
        start_time = time.time()

        # Get input values
        smiles_str = smiles_input.value.strip() if smiles_input.value else ""
        taxon_input_str = taxon_input.value.strip()

        # Determine search mode based on inputs
        # 1. Both present: use both (SMILES search filtered by taxon)
        # 2. SMILES only (taxon is empty or "*"): use SMILES only
        # 3. Taxon only (SMILES empty): use taxon only

        use_smiles = bool(smiles_str)
        use_taxon = bool(taxon_input_str and taxon_input_str != "*")

        # Initialize all return values for all paths
        qid = None
        taxon_warning = None
        lazy_results = None

        if use_smiles and use_taxon:
            # Both present - search by structure within taxon
            spinner_message = (
                f"Searching for SMILES '{smiles_str}' in {taxon_input_str}"
            )
        elif use_smiles:
            # SMILES only
            spinner_message = f"Searching for SMILES: {smiles_str}"
        else:
            # Taxon only
            if taxon_input_str == "*":
                spinner_message = "Searching all taxa ..."
            else:
                spinner_message = f"Searching for: {taxon_input_str}"

        with mo.status.spinner(title=spinner_message):
            # Resolve taxon if using taxon (either alone or with SMILES)
            if use_taxon or (not use_smiles and taxon_input_str):
                taxon_input_str = taxon_input.value.strip()
                qid, taxon_warning = resolve_taxon_to_qid(taxon_input_str)
                if not qid:
                    mo.stop(
                        True,
                        mo.callout(
                            mo.md(
                                f"**Taxon not found**\n\n"
                                f"Could not find '{taxon_input_str}' in Wikidata.\n\n"
                                f"**Suggestions:**\n"
                                f"- Check spelling (scientific names are case-sensitive)\n"
                                f"- Try a different taxonomic level (e.g., genus instead of species)\n"
                                f"- Use a Wikidata QID directly (e.g., Q157115)",
                            ),
                            kind="warn",
                        ).style(
                            style={
                                "overflow-wrap": "anywhere",
                            },
                        ),
                    )

            try:
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
                lazy_results = query_wikidata(
                    qid=qid if qid else "",
                    year_start=year_start.value if year_filter.value else None,
                    year_end=year_end.value if year_filter.value else None,
                    mass_min=mass_min.value if mass_filter.value else None,
                    mass_max=mass_max.value if mass_filter.value else None,
                    formula_filters=formula_filt,
                    smiles=smiles_str,
                    smiles_search_type=smiles_search_type.value,
                    smiles_threshold=smiles_threshold.value,
                )
            except Exception as e:
                mo.stop(
                    True,
                    mo.callout(
                        mo.md(f"**Query Error:** {str(e)}"),
                        kind="danger",
                    ).style(
                        style={
                            "overflow-wrap": "anywhere",
                        },
                    ),
                )
        elapsed = round(time.time() - start_time, 2)
        _ = mo.md(f"Query executed in **{elapsed}s**")

    return qid, lazy_results, taxon_warning


@app.function
def get_counts(lazy_df: pl.LazyFrame) -> dict:
    """
    Get counts without materializing full DataFrame.

    Only materializes 1 row with aggregations.
    """

    counts_lazy = lazy_df.select(
        [
            pl.col("compound").n_unique().alias("n_compounds"),
            pl.col("taxon").n_unique().alias("n_taxa"),
            pl.col("reference").n_unique().alias("n_refs"),
            pl.len().alias("n_entries"),
        ],
    )

    # Materialize ONLY the counts (1 row)
    counts_df = counts_lazy.collect()

    if counts_df.is_empty():
        return {"n_compounds": 0, "n_taxa": 0, "n_refs": 0, "n_entries": 0}

    return {
        "n_compounds": int(counts_df["n_compounds"][0]),
        "n_taxa": int(counts_df["n_taxa"][0]),
        "n_refs": int(counts_df["n_refs"][0]),
        "n_entries": int(counts_df["n_entries"][0]),
    }


@app.cell
def get_counts_lazy(lazy_results):
    """Get counts without materializing full dataset."""

    if lazy_results is None:
        counts = None
    else:
        # *** ONLY MATERIALIZES 1 ROW (4 integers) ***
        counts = get_counts(lazy_results)
    return counts


@app.cell
def display_summary(
    br_state,
    c_max,
    c_min,
    cl_state,
    download_ui,
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
    qid,
    query_hash,
    result_hash,
    lazy_results,
    counts,
    run_button,
    s_max,
    s_min,
    search_params,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    taxon_warning,
    year_end,
    year_filter,
    year_start,
):
    # Display summary if either button was clicked or auto-run from URL
    if (not run_button.value and not search_params.auto_run) or lazy_results is None:
        summary_and_downloads = mo.Html("")
    elif counts["n_compounds"] == 0:
        # Show no compounds message, and taxon warning if present
        parts = []
        if taxon_warning:
            parts.append(
                mo.callout(taxon_warning, kind="warn").style(
                    style={
                        "overflow-wrap": "anywhere",
                    },
                ),
            )

        # Handle wildcard case
        if qid == "*":
            parts.append(
                mo.callout(
                    mo.md(
                        "No compounds found for **all taxa** with the current filters.",
                    ),
                    kind="warn",
                ).style(
                    style={
                        "overflow-wrap": "anywhere",
                    },
                ),
            )
        else:
            parts.append(
                mo.callout(
                    mo.md(
                        f"No compounds found for **{taxon_input.value}** ([{qid}]({scholia_url(qid)})) with the current filters.",
                    ),
                    kind="warn",
                ).style(
                    style={
                        "overflow-wrap": "anywhere",
                    },
                ),
            )
        summary_and_downloads = mo.vstack(parts) if len(parts) > 1 else parts[0]
    else:
        n_compounds = counts["n_compounds"]
        n_taxa = counts["n_taxa"]
        n_refs = counts["n_refs"]
        n_entries = counts["n_entries"]

        # Results header (on its own line)
        results_header = mo.md("## Results")

        # Taxon info
        if qid == "*":
            taxon_info = "All taxa"
        else:
            taxon_info = f"{taxon_input.value} [{qid}]({scholia_url(qid)})"

        # Add SMILES search info if present
        if smiles_input.value and smiles_input.value.strip():
            _smiles_str = smiles_input.value.strip()
            search_type = smiles_search_type.value

            if search_type == "similarity":
                threshold_val = smiles_threshold.value
                smiles_info = f"SMILES: `{_smiles_str}` ({search_type}, threshold: {threshold_val})"
            else:
                smiles_info = f"SMILES: `{_smiles_str}` ({search_type})"

            combined_info = f"{taxon_info} - {smiles_info}"
        else:
            combined_info = taxon_info

        # Search info
        search_info_display = mo.md(
            f"**{combined_info}**\n"
            f"**Hashes:**\n"
            f"\t*Query*: `{query_hash}`"
            f"\t*Results*: `{result_hash}`",
        ).style(
            style={
                "overflow-wrap": "anywhere",
            },
        )

        # Stats cards - use list comprehension for DRY
        stats_data = [
            (n_compounds, "Compound"),
            (n_taxa, "Taxon"),
            (n_refs, "Reference"),
            (n_entries, "Entry"),
        ]
        stats_cards = mo.hstack(
            [
                mo.stat(
                    value=f"{n:,}",
                    label=f"{pluralize(name, n, irregular=PLURAL_MAP)}",
                    bordered=False,
                )
                for n, name in stats_data
            ],
            gap=0,
            justify="start",
            wrap=True,
        ).style(
            style={
                "overflow-wrap": "anywhere",
            },
        )

        search_and_stats = mo.vstack(
            [
                search_info_display,
                stats_cards,
            ],
            gap=2,
        )

        # Build API URL for sharing
        api_url = build_api_url(
            taxon=taxon_input.value,
            smiles=smiles_input.value,
            smiles_search_type=smiles_search_type.value,
            smiles_threshold=smiles_threshold.value,
            mass_filter=mass_filter.value,
            mass_min=mass_min.value,
            mass_max=mass_max.value,
            year_filter=year_filter.value,
            year_start=year_start.value,
            year_end=year_end.value,
            formula_filter=formula_filter.value,
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

        # Display shareable URL if parameters exist
        if api_url:
            url_display = mo.md(
                f"""
                **Shareable URL**

                Copy and append this to your notebook URL to share this exact search:
                ```
                {api_url}
                ```
                """,
            )
            api_url_section = mo.accordion(
                {"Share this search": url_display},
                multiple=False,
            )
        else:
            api_url_section = mo.Html("")

        # Build summary section with all parts
        summary_parts = [results_header, search_and_stats]

        if api_url:
            summary_parts.append(api_url_section)

        if taxon_warning:
            summary_parts.append(
                mo.callout(taxon_warning, kind="warn").style(
                    style={
                        "overflow-wrap": "anywhere",
                    },
                ),
            )

        # Stack summary and downloads vertically
        summary_and_downloads = mo.vstack(
            [mo.vstack(summary_parts), download_ui],
            gap=3,
        )

    summary_and_downloads
    return


@app.cell
def generate_results(
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
    qid,
    lazy_results,
    counts,
    run_button,
    s_max,
    s_min,
    search_params,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    # Replace previous generation logic: build UI but DO NOT display inline
    if (not run_button.value and not search_params.auto_run) or lazy_results is None:
        download_ui = mo.Html("")
        tables_ui = mo.Html("")
        ui_is_large_dataset = False
        taxon_name = taxon_input.value
        active_filters = {}
        csv_generate_button = None
        json_generate_button = None
        rdf_generate_button = None
        csv_generation_data = None
        json_generation_data = None
        rdf_generation_data = None
        query_hash = None
        result_hash = None
    elif counts["n_entries"] == 0:
        download_ui = mo.callout(
            mo.md("No compounds match your search criteria."),
            kind="neutral",
        ).style(
            style={
                "overflow-wrap": "anywhere",
            },
        )
        tables_ui = mo.Html("")
        ui_is_large_dataset = False
        taxon_name = taxon_input.value
        active_filters = {}
        csv_generate_button = None
        json_generate_button = None
        rdf_generate_button = None
        csv_generation_data = None
        json_generation_data = None
        rdf_generation_data = None
        query_hash = None
        result_hash = None
    else:
        # Build filters for metadata (needed for hash computation)
        _formula_filt = None
        if formula_filter.value:
            _formula_filt = create_filters(
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
        active_filters = build_active_filters_dict(
            mass_filter_active=mass_filter.value,
            mass_min_val=mass_min.value if mass_filter.value else None,
            mass_max_val=mass_max.value if mass_filter.value else None,
            year_filter_active=year_filter.value,
            year_start_val=year_start.value if year_filter.value else None,
            year_end_val=year_end.value if year_filter.value else None,
            formula_filters=_formula_filt,
            smiles=smiles_input.value,
            smiles_search_type=smiles_search_type.value,
            smiles_threshold=smiles_threshold.value,
        )

        # Compute hashes for provenance using centralized helper
        query_hash, result_hash = compute_provenance_hashes(
            qid,
            taxon_input.value,
            active_filters,
            lazy_results,
        )

        # Check if this is a large dataset BEFORE preparing export dataframes
        # This avoids creating copies of data in memory for large datasets
        taxon_name = taxon_input.value
        ui_is_large_dataset = counts["n_entries"] > CONFIG["table_row_limit"]

        # For large datasets, defer export dataframe preparation to download time
        if ui_is_large_dataset:
            # Store reference to original df - export prep happens on-demand
            export_df = None
            export_df_rdf = None
        else:
            # For small datasets, prepare export dataframes immediately
            export_df = prepare_export_dataframe(lazy_results, include_rdf_ref=False)
            export_df_rdf = prepare_export_dataframe(lazy_results, include_rdf_ref=True)

        # Create metadata using already-built active_filters
        metadata = create_export_metadata(
            lazy_results if ui_is_large_dataset else export_df,
            counts,
            taxon_input.value,
            qid,
            active_filters,
            query_hash,
            result_hash,
        )
        metadata_json = json.dumps(metadata, indent=2)
        citation_text = create_citation_text(taxon_input.value)
        # Display table data (apply row limit & depiction logic)
        total_rows = counts["n_entries"]
        if total_rows > CONFIG["table_row_limit"]:
            limited_df = lazy_results.head(CONFIG["table_row_limit"])
            display_note = mo.callout(
                mo.md(
                    f"**Large Dataset Optimization**\n\n"
                    f"Your search returned **{total_rows:,} rows**. For optimal performance:\n"
                    f"- Displaying first **{CONFIG['table_row_limit']:,} rows** in table view\n"
                    f"- Downloads are generated on-demand (click Generate buttons)\n"
                    f"- Export view disabled for large datasets",
                ),
                kind="info",
            ).style(
                style={
                    "overflow-wrap": "anywhere",
                },
            )
        else:
            display_note = mo.Html("")
            limited_df = lazy_results

        # Build display DataFrame with HTML-formatted columns
        display_df = build_display_dataframe(limited_df)

        # Try the nice mo.ui.table first, fall back to df.style if it fails
        try:
            display_table = mo.ui.table(
                data=display_df,
                format_mapping={
                    "Compound Depiction": wrap_image,
                    "Reference DOI": wrap_html,
                    "Compound QID": wrap_html,
                    "Taxon QID": wrap_html,
                    "Reference QID": wrap_html,
                    "Statement": wrap_html,
                },
                selection=None,
                page_size=CONFIG["page_size_default"],
            )
        except Exception:
            display_table = display_df.style

        # Export table: only show for smaller datasets
        if not ui_is_large_dataset and export_df is not None:
            try:
                export_table_ui = mo.ui.table(
                    data=export_df,
                    selection=None,
                    page_size=CONFIG["page_size_export"],
                )
            except Exception:
                export_table_ui = export_df.style
        else:
            export_table_ui = mo.callout(
                mo.md(
                    f"**Large Dataset ({total_rows:,} rows)**\n\n"
                    f"Export table view is disabled for datasets over {CONFIG['table_row_limit']} rows "
                    f"to ensure smooth performance.\n\n"
                    f"Use the download buttons above to get your data in CSV, JSON, or RDF format.",
                ),
                kind="info",
            ).style(
                style={
                    "overflow-wrap": "anywhere",
                },
            )

        # Download buttons generation
        if ui_is_large_dataset:
            # Lazy generation buttons
            csv_generate_button = mo.ui.run_button(label="Generate CSV")
            json_generate_button = mo.ui.run_button(label="Generate JSON")
            rdf_generate_button = mo.ui.run_button(label="Generate RDF/Turtle")
            buttons = [
                csv_generate_button,
                json_generate_button,
                rdf_generate_button,
            ]
            # Store results_df ONCE - all exports share the same data reference
            shared_generation_data = {
                "results_df": lazy_results,
                "taxon_input": taxon_input.value,
                "qid": qid,
                "active_filters": active_filters,
                "lazy": True,
            }
            csv_generation_data = shared_generation_data
            json_generation_data = shared_generation_data
            rdf_generation_data = shared_generation_data
            # Metadata download
            metadata_button = mo.download(
                data=metadata_json,
                filename=generate_filename(
                    taxon_input.value,
                    "json",
                    prefix="lotus_metadata",
                    filters=active_filters,
                ),
                label="Metadata",
                mimetype="application/json",
            )
            download_ui = mo.vstack(
                [
                    mo.md("### Download Data"),
                    mo.hstack(buttons, gap=2, wrap=True),
                    metadata_button,
                ],
            )
        else:
            csv_generate_button = None
            json_generate_button = None
            rdf_generate_button = None
            csv_generation_data = None
            json_generation_data = None
            rdf_generation_data = None
            buttons = [
                create_download_button(
                    export_df.collect().write_csv(),
                    generate_filename(taxon_input.value, "csv", filters=active_filters),
                    "CSV",
                    "text/csv",
                ),
                create_download_button(
                    export_df.collect().write_json(),
                    generate_filename(
                        taxon_input.value,
                        "json",
                        filters=active_filters,
                    ),
                    "JSON",
                    "application/json",
                ),
                create_download_button(
                    export_to_rdf_turtle(
                        export_df_rdf,
                        taxon_input.value,
                        qid,
                        active_filters,
                    ),
                    generate_filename(taxon_input.value, "ttl", filters=active_filters),
                    "RDF/Turtle",
                    "text/turtle",
                ),
                mo.download(
                    data=metadata_json,
                    filename=generate_filename(
                        taxon_input.value,
                        "json",
                        prefix="lotus_metadata",
                        filters=active_filters,
                    ),
                    label="Metadata",
                    mimetype="application/json",
                ),
            ]
            download_ui = mo.vstack(
                [mo.md("### Download Data"), mo.hstack(buttons, gap=2, wrap=True)],
            )
        tables_ui = mo.vstack(
            [
                mo.md("### Browse Data"),
                display_note,
                mo.ui.tabs(
                    {
                        "Display": display_table,
                        "Export View": export_table_ui,
                        "Citation": mo.md(citation_text),
                        "Metadata": mo.md(f"```json\n{metadata_json}\n```"),
                    },
                ).style(
                    style={
                        "overflow-wrap": "anywhere",
                    },
                ),
            ],
        )


@app.cell
def generate_downloads(
    csv_generate_button,
    csv_generation_data,
    json_generate_button,
    json_generation_data,
    rdf_generate_button,
    rdf_generation_data,
    taxon_name,
    ui_is_large_dataset,
):
    """Handle lazy generation of downloads for large datasets."""

    def get_export_df(generation_data, include_rdf_ref=False):
        """Get or prepare export dataframe on-demand."""
        if generation_data.get("lazy"):
            # Prepare export dataframe now (deferred from results cell)
            return prepare_export_dataframe(
                generation_data["results_df"],
                include_rdf_ref=include_rdf_ref,
            )
        else:
            return generation_data["export_df"]

    def create_lazy_download_ui(
        generate_button,
        generation_data,
        format_name: str,
        format_ext: str,
        data_generator_fn,
        base_mimetype: str,
        is_large: bool,
        include_rdf_ref: bool = False,
    ):
        """Generic lazy download UI generator."""
        if not is_large or generate_button is None or not generate_button.value:
            return mo.Html("")

        with mo.status.spinner(title=f"Generating {format_name} format..."):
            # Prepare export dataframe on-demand
            export_df = get_export_df(generation_data, include_rdf_ref)
            raw_data = data_generator_fn(export_df, generation_data)

            # Convert to bytes if needed
            raw_bytes = (
                raw_data if isinstance(raw_data, bytes) else raw_data.encode("utf-8")
            )

            # Compress if needed - returns (data, was_compressed)
            compressed_data, was_compressed = compress_if_large(
                raw_bytes,
                CONFIG["download_embed_threshold_bytes"],
            )

            final_filename = generate_filename(
                taxon_name,
                format_ext,
                filters=generation_data["active_filters"],
            )
            if was_compressed:
                final_filename += ".gz"
            final_mimetype = "application/gzip" if was_compressed else None

        mimetype = final_mimetype if final_mimetype else base_mimetype
        display_label = f"Download {format_name}"

        # Use standard mo.download
        download_button = mo.download(
            data=compressed_data,
            filename=final_filename,
            label=display_label,
            mimetype=mimetype,
        )

        return mo.vstack(
            [
                mo.callout(
                    mo.md(
                        f"**{format_name} Ready** - {len(export_df.collect()):,} entries"
                        + (
                            " (compressed)"
                            if final_mimetype == "application/gzip"
                            else ""
                        ),
                    ),
                    kind="success",
                ).style(
                    style={
                        "overflow-wrap": "anywhere",
                    },
                ),
                download_button,
            ],
        )

    # CSV generation
    csv_download_ui = create_lazy_download_ui(
        csv_generate_button,
        csv_generation_data,
        "CSV",
        "csv",
        lambda df, d: df.collect().write_csv(),
        "text/csv",
        ui_is_large_dataset,
    )

    # JSON generation
    json_download_ui = create_lazy_download_ui(
        json_generate_button,
        json_generation_data,
        "JSON",
        "json",
        lambda df, d: df.collect().write_json(),
        "application/json",
        ui_is_large_dataset,
    )

    # RDF generation (needs include_rdf_ref=True)
    def _rdf_generator(df, d):
        return export_to_rdf_turtle(df, d["taxon_input"], d["qid"], d["active_filters"])

    rdf_download_ui = create_lazy_download_ui(
        rdf_generate_button,
        rdf_generation_data,
        "RDF/Turtle",
        "ttl",
        _rdf_generator,
        "text/turtle",
        ui_is_large_dataset,
        include_rdf_ref=True,
    )

    # Show all generated downloads
    _out = mo.vstack([csv_download_ui, json_download_ui, rdf_download_ui], gap=2)
    _out
    return


@app.cell
def ui_tables(tables_ui):
    tables_ui
    return


@app.cell
def ui_params():
    # URL parameter detection and state initialization using SearchParams dataclass
    _query_params = mo.query_params()
    # Convert QueryParams to dict for SearchParams
    _url_params = {k: _query_params[k] for k in _query_params} if _query_params else {}
    search_params = SearchParams.from_url_params(_url_params)

    # Display auto-search message if URL parameters detected
    if search_params.auto_run:
        _ = mo.md(
            f"**Auto-executing search for:** {search_params.taxon if search_params.taxon else search_params.smiles}",
        )
    return (search_params,)


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


@app.function
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # CLI mode - extract and reuse app.setup functions
        import argparse
        import gzip
        from pathlib import Path

        # Parse CLI arguments
        parser = argparse.ArgumentParser(description="Export LOTUS data")
        parser.add_argument("export")
        parser.add_argument("--taxon", help="Taxon name or QID")
        parser.add_argument("--output", "-o", help="Output file")
        parser.add_argument(
            "--format",
            "-f",
            choices=["csv", "json", "ttl"],
            default="csv",
        )
        parser.add_argument("--year-start", type=int, help="Minimum publication year")
        parser.add_argument("--year-end", type=int, help="Maximum publication year")
        parser.add_argument("--mass-min", type=float, help="Minimum molecular mass")
        parser.add_argument("--mass-max", type=float, help="Maximum molecular mass")

        # Molecular formula filters
        parser.add_argument(
            "--formula",
            help="Exact molecular formula (e.g., C15H10O5)",
        )
        parser.add_argument("--c-min", type=int, help="Minimum carbon atoms")
        parser.add_argument("--c-max", type=int, help="Maximum carbon atoms")
        parser.add_argument("--h-min", type=int, help="Minimum hydrogen atoms")
        parser.add_argument("--h-max", type=int, help="Maximum hydrogen atoms")
        parser.add_argument("--n-min", type=int, help="Minimum nitrogen atoms")
        parser.add_argument("--n-max", type=int, help="Maximum nitrogen atoms")
        parser.add_argument("--o-min", type=int, help="Minimum oxygen atoms")
        parser.add_argument("--o-max", type=int, help="Maximum oxygen atoms")

        parser.add_argument(
            "--smiles",
            help="SMILES string for chemical structure search",
        )
        parser.add_argument(
            "--smiles-search-type",
            choices=["exact", "substructure", "similarity"],
            help="Type of SMILES search (exact, substructure, or similarity)",
        )
        parser.add_argument(
            "--smiles-threshold",
            type=float,
            default=0.8,
            help="Similarity threshold (0.0-1.0, default: 0.8)",
        )
        parser.add_argument(
            "--compress",
            action="store_true",
            help="Compress output with gzip",
        )
        parser.add_argument(
            "--show-metadata",
            action="store_true",
            help="Display FAIR-compliant dataset metadata and exit (no data export)",
        )
        parser.add_argument(
            "--export-metadata",
            action="store_true",
            help="Export metadata as JSON file alongside data (creates <output>.metadata.json)",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Verbose output",
        )
        args = parser.parse_args()

        try:
            with open(__file__, encoding="utf-8") as f:
                file_content = f.read()

            # Strategy: Extract app.setup block AND all @app.function decorated functions
            # This ensures we get imports, constants, and all the actual functions
            setup_start = file_content.find("with app.setup:")
            if setup_start == -1:
                raise RuntimeError("Could not find app.setup block")

            setup_start += len("with app.setup:\n")
            next_app_cell = file_content.find("\n@app.", setup_start)

            if next_app_cell == -1:
                raise RuntimeError("Could not find end of app.setup block")

            setup_block = file_content[setup_start:next_app_cell]

            # Dedent app.setup (remove 4-space indentation)
            setup_lines = []
            for line in setup_block.split("\n"):
                if line.startswith("    "):
                    setup_lines.append(line[4:])
                elif line.strip() == "":
                    setup_lines.append("")
                else:
                    break

            # Extract all @app.class_definition blocks (dataclasses like ElementRange, FormulaFilters, SearchParams)

            class_pattern = r"@app\.class_definition\s*\n@dataclass"
            class_blocks = []

            for match in re.finditer(class_pattern, file_content):
                class_start = match.start()

                # Find where this class ends (next @app. or def main)
                next_decorator = file_content.find("\n@app.", class_start + 1)
                next_main = file_content.find("\ndef main", class_start)

                if next_decorator != -1:
                    class_end = next_decorator
                elif next_main != -1:
                    class_end = next_main
                else:
                    class_end = len(file_content)

                # Extract class definition (skip the @app.class_definition decorator)
                dataclass_start = file_content.find("@dataclass", class_start)
                class_block = file_content[dataclass_start:class_end].strip()
                class_blocks.append(class_block)

            # Extract all @app.function blocks (they contain the actual functions)
            function_pattern = r"@app\.function\s*\ndef\s+(\w+)"
            function_blocks = []

            # Find all @app.function decorated functions
            for match in re.finditer(function_pattern, file_content):
                func_name = match.group(1)
                func_start = match.start()

                # Find where this function ends (next @app. or if __name__)
                next_decorator = file_content.find("\n@app.", func_start + 1)
                next_main = file_content.find("\nif __name__", func_start)

                if next_decorator != -1:
                    func_end = next_decorator
                elif next_main != -1:
                    func_end = next_main
                else:
                    func_end = len(file_content)

                # Extract function definition (skip the @app.function decorator)
                func_def_start = file_content.find("def " + func_name, func_start)
                func_block = file_content[func_def_start:func_end].strip()
                function_blocks.append(func_block)

            # Combine setup, classes, and functions into executable code
            combined_code = (
                "\n".join(setup_lines)
                + "\n\n"
                + "\n\n".join(class_blocks)
                + "\n\n"
                + "\n\n".join(function_blocks)
            )

            # Execute in isolated namespace
            namespace = {}
            exec(combined_code, namespace)

            if args.taxon is None:
                args.taxon = "*"

            if args.verbose:
                print(f"Querying LOTUS data for: {args.taxon}", file=sys.stderr)

            # Resolve taxon using the REAL function
            qid, warning = resolve_taxon_to_qid(args.taxon)
            if not qid:
                print(f"[x] Taxon not found: {args.taxon}", file=sys.stderr)
                sys.exit(1)

            # Improved verbose logging
            if args.verbose:
                print("Search Configuration:", file=sys.stderr)
                print(f"   Taxon: {args.taxon} (QID: {qid})", file=sys.stderr)

                if args.smiles:
                    print(f"   SMILES: {args.smiles}", file=sys.stderr)
                    print(
                        f"   Search Type: {args.smiles_search_type or 'substructure'}",
                        file=sys.stderr,
                    )
                    if args.smiles_search_type == "similarity":
                        print(
                            f"   Similarity Threshold: {args.smiles_threshold}",
                            file=sys.stderr,
                        )

                filters_applied = []
                if args.year_start:
                    filters_applied.append(f"year >= {args.year_start}")
                if args.year_end:
                    filters_applied.append(f"year <= {args.year_end}")
                if args.mass_min:
                    filters_applied.append(f"mass >= {args.mass_min}")
                if args.mass_max:
                    filters_applied.append(f"mass <= {args.mass_max}")
                if args.formula:
                    filters_applied.append(f"formula = {args.formula}")
                elem_args = [
                    args.c_min,
                    args.c_max,
                    args.h_min,
                    args.h_max,
                    args.n_min,
                    args.n_max,
                    args.o_min,
                    args.o_max,
                ]
                if any(elem_args):
                    filters_applied.append("element ranges")
                if filters_applied:
                    print(
                        f"   Filters: {', '.join(filters_applied)}",
                        file=sys.stderr,
                    )
                print(file=sys.stderr)

            # Build formula filters if any formula arguments provided
            formula_filt = None
            elem_args = [
                args.c_min,
                args.c_max,
                args.h_min,
                args.h_max,
                args.n_min,
                args.n_max,
                args.o_min,
                args.o_max,
            ]
            if args.formula or any(elem_args):
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

            # Query using the REAL function with all arguments
            df = query_wikidata(
                qid=qid,
                year_start=args.year_start,
                year_end=args.year_end,
                mass_min=args.mass_min,
                mass_max=args.mass_max,
                formula_filters=formula_filt,
                smiles=args.smiles,
                smiles_search_type=args.smiles_search_type or "substructure",
                smiles_threshold=args.smiles_threshold,
            ).collect()

            if df.is_empty():
                print("[x] No data found", file=sys.stderr)
                sys.exit(1)

            if args.verbose:
                print("Query Results:", file=sys.stderr)
                print(f"   Total entries: {len(df):,}", file=sys.stderr)

                # Show unique counts if columns exist
                if "compound" in df.collect_schema().names():
                    unique_compounds = df.select(pl.col("compound")).n_unique()
                    print(
                        f"   Unique compounds: {unique_compounds:,}",
                        file=sys.stderr,
                    )
                if "taxon" in df.collect_schema().names():
                    unique_taxa = df.select(pl.col("taxon")).n_unique()
                    print(f"   Unique taxa: {unique_taxa:,}", file=sys.stderr)
                if "reference" in df.collect_schema().names():
                    unique_refs = df.select(pl.col("reference")).n_unique()
                    print(
                        f"   Unique references: {unique_refs:,}",
                        file=sys.stderr,
                    )
                print(file=sys.stderr)  # Empty line for readability

            # Build filters dict in the same format as the UI (needed for metadata and provenance)

            filters = {}

            # Year filter
            if args.year_start or args.year_end:
                filters["publication_year"] = {}
                if args.year_start:
                    filters["publication_year"]["min"] = args.year_start
                if args.year_end:
                    filters["publication_year"]["max"] = args.year_end

            # Mass filter
            if args.mass_min or args.mass_max:
                filters["molecular_mass"] = {}
                if args.mass_min:
                    filters["molecular_mass"]["min"] = args.mass_min
                if args.mass_max:
                    filters["molecular_mass"]["max"] = args.mass_max

            # Chemical structure filter
            if args.smiles:
                filters["chemical_structure"] = {
                    "smiles": args.smiles,
                    "search_type": args.smiles_search_type or "substructure",
                }
                if args.smiles_search_type == "similarity":
                    filters["chemical_structure"]["similarity_threshold"] = (
                        args.smiles_threshold
                    )

            # Molecular formula filter
            if formula_filt and formula_filt.is_active():
                filters["molecular_formula"] = serialize_filters(formula_filt)

            # Compute hashes for provenance using centralized helper
            query_hash, result_hash = compute_provenance_hashes(
                qid,
                args.taxon,
                filters if filters else None,
                df,
            )

            # Show metadata mode - use the REAL create_export_metadata function
            if args.show_metadata:
                # Use the REAL metadata function from app.setup with provenance hashes
                metadata = create_export_metadata(
                    df,
                    {"n_entries": len(df)},
                    args.taxon,
                    qid,
                    filters if filters else None,
                    query_hash=query_hash,
                    result_hash=result_hash,
                )
                print(json.dumps(metadata, indent=2))

            # Export data using REAL functions
            if args.format == "csv":
                # CSV export: exclude ref column (statement included for transparency)
                export_df = prepare_export_dataframe(
                    df,
                    include_rdf_ref=False,
                )
                data = export_df.write_csv().encode("utf-8")
            elif args.format == "json":
                # JSON export: exclude ref column (statement included for transparency)
                export_df = prepare_export_dataframe(
                    df,
                    include_rdf_ref=False,
                )
                data = export_df.write_json().encode("utf-8")
            elif args.format == "ttl":
                # RDF export: include ref column for full provenance
                export_df = prepare_export_dataframe(df, include_rdf_ref=True)
                data = export_to_rdf_turtle(
                    export_df,
                    args.taxon,
                    qid,
                    filters if filters else None,
                ).encode("utf-8")
            else:
                print(f"[x] Unknown format: {args.format}", file=sys.stderr)
                sys.exit(1)

            # Compress if requested
            if args.compress:
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                    gz.write(data)
                data = buffer.getvalue()

            # Write output
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(data)
                if args.verbose:
                    print(
                        f"[+] Exported {len(data):,} bytes to: {output_path}",
                        file=sys.stderr,
                    )

                # Export metadata if requested
                if args.export_metadata:
                    metadata = create_export_metadata(
                        df,
                        {"n_entries": len(df)},
                        args.taxon,
                        qid,
                        filters if filters else None,
                        query_hash=query_hash,
                        result_hash=result_hash,
                    )
                    metadata_path = output_path.with_suffix(
                        output_path.suffix + ".metadata.json",
                    )
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    if args.verbose:
                        print(
                            f"[+] Exported metadata to: {metadata_path}",
                            file=sys.stderr,
                        )
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
