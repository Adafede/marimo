# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "astral==3.2",
#     "polars==1.35.1",
#     "pyarrow==22.0.0",
#     "sparqlwrapper==2.0.0",
# ]
# [tool.marimo.display]
# theme = "system"
# ///

"""
LOTUS Wikidata Explorer

Copyright (C) 2025 Adriano Rutz

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

__generated_with = "0.17.0"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import polars as pl
    import re
    import time
    from dataclasses import dataclass, field
    from datetime import datetime
    from functools import lru_cache
    from typing import Optional, Dict, Any, Tuple
    from urllib.parse import quote
    from SPARQLWrapper import SPARQLWrapper, JSON

    # ====================================================================
    # CONFIGURATION
    # ====================================================================

    CONFIG = {
        "cdk_base": "https://www.simolecule.com/cdkdepict/depict/cot/svg",
        "color_hyperlink": "#006699",
        "max_retries": 3,
        "page_size_default": 15,
        "page_size_export": 25,
        "query_limit": 1_000_000,
        "retry_backoff": 2,
        "user_agent": "LOTUS Explorer/0.0.1",
    }

    # Wikidata URLs (constants)
    WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
    WIKIDATA_WIKI_PREFIX = "https://www.wikidata.org/wiki/"

    # Shared SPARQL instance
    SPARQL = SPARQLWrapper("https://query.wikidata.org/sparql")
    SPARQL.setReturnFormat(JSON)
    SPARQL.addCustomHttpHeader("User-Agent", CONFIG["user_agent"])

    # Subscript translation map (constant for performance)
    SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    # Regex pattern for molecular formula parsing (compiled once)
    FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")

    # Pluralization map (constant)
    PLURAL_MAP = {
        "Entry": "Entries",
        "entry": "entries",
        "Taxon": "Taxa",
        "taxon": "taxa",
    }

    # ====================================================================
    # DATA CLASSES (SOLID - Single Responsibility)
    # ====================================================================

    @dataclass(frozen=True)
    class ElementRange:
        """Range for element count in molecular formula."""

        min_val: Optional[int] = None
        max_val: Optional[int] = None

        def is_active(self) -> bool:
            """Check if range filter is active."""
            return self.min_val is not None or self.max_val is not None

        def matches(self, count: int) -> bool:
            """Check if count is within range."""
            if not self.is_active():
                return True
            if self.min_val is not None and count < self.min_val:
                return False
            if self.max_val is not None and count > self.max_val:
                return False
            return True

    @dataclass(frozen=True)
    class FormulaFilters:
        """Molecular formula filtering criteria."""

        exact_formula: Optional[str] = None
        c: ElementRange = field(default_factory=ElementRange)
        h: ElementRange = field(default_factory=ElementRange)
        n: ElementRange = field(default_factory=ElementRange)
        o: ElementRange = field(default_factory=ElementRange)
        p: ElementRange = field(default_factory=ElementRange)
        s: ElementRange = field(default_factory=ElementRange)
        f_state: str = "allowed"
        cl_state: str = "allowed"
        br_state: str = "allowed"
        i_state: str = "allowed"

        def is_active(self) -> bool:
            """Check if any filter is active."""
            if self.exact_formula and self.exact_formula.strip():
                return True
            if any(
                r.is_active() for r in [self.c, self.h, self.n, self.o, self.p, self.s]
            ):
                return True
            if any(
                s != "allowed"
                for s in [self.f_state, self.cl_state, self.br_state, self.i_state]
            ):
                return True
            return False


@app.function
def build_taxon_search_query(taxon_name: str) -> str:
    """Build SPARQL query to find taxa by scientific name. Returns up to 10 results."""
    return f"""
    SELECT ?taxon ?taxon_name WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{taxon_name}";
                        mwapi:language "mul".
        ?taxon wikibase:apiOutputItem mwapi:item.
        ?num wikibase:apiOrdinal true.
      }}
      ?taxon wdt:P225 ?taxon_name.
    }}
    """


@app.function
def build_compounds_query(qid: str) -> str:
    return f"""
    SELECT DISTINCT ?compound ?compoundLabel ?compound_inchikey ?compound_smiles_iso ?compound_smiles_conn  ?compound_mass ?compound_formula
                   ?taxon_name ?taxon ?ref_title ?ref_doi ?ref_qid ?ref_date
    WHERE {{
      ?taxon (wdt:P171*) wd:{qid};
             wdt:P225 ?taxon_name. 
      ?compound wdt:P235 ?compound_inchikey;
                wdt:P233 ?compound_smiles_conn;
                p:P703 ?statement.
      ?statement ps:P703 ?taxon;
                 prov:wasDerivedFrom ?ref.
      ?ref pr:P248 ?ref_qid.
      OPTIONAL {{ ?compound wdt:P2017 ?compound_smiles_iso. }}
      OPTIONAL {{ ?compound wdt:P2067 ?compound_mass. }}
      OPTIONAL {{ ?compound wdt:P274 ?compound_formula. }}
      OPTIONAL {{
        SERVICE <https://query-scholarly.wikidata.org/sparql> {{
          ?ref_qid wdt:P1476 ?ref_title;
                   wdt:P356 ?ref_doi;
                   wdt:P577 ?ref_date.
        }}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {CONFIG["query_limit"]}
    """


@app.function
@lru_cache(maxsize=128)
def execute_sparql(
    query: str, max_retries: int = CONFIG["max_retries"]
) -> Dict[str, Any]:
    """
    Execute SPARQL query with retry logic and exponential backoff.

    Cached to improve performance and reduce load on Wikidata servers.
    Cache key is the query string itself.
    """
    for attempt in range(max_retries):
        try:
            SPARQL.setQuery(query)
            return SPARQL.query().convert()
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Query failed after {max_retries} attempts: {str(e)}")
            wait_time = CONFIG["retry_backoff"] * (2**attempt)
            time.sleep(wait_time)
    # This line should never be reached, but satisfies type checker
    raise Exception("Unexpected error in execute_sparql")


@app.function
@lru_cache(maxsize=512)
def extract_qid(url: str) -> str:
    """Extract QID from Wikidata entity URL. Cached for performance."""
    return url.replace(WIKIDATA_ENTITY_PREFIX, "")


@app.function
@lru_cache(maxsize=1024)
def create_structure_image_url(smiles: str) -> str:
    if not smiles:
        return "https://via.placeholder.com/120x120?text=No+SMILES"
    encoded_smiles = quote(smiles)
    return f"{CONFIG['cdk_base']}?smi={encoded_smiles}&annotate=cip"


@app.function
def build_taxon_details_query(qids: list) -> str:
    """Build SPARQL query to fetch taxon details (description and parent taxon)."""
    qids_str = " ".join(f"wd:{qid}" for qid in qids)
    return f"""
    SELECT ?taxon ?taxonLabel ?taxonDescription ?taxon_parentLabel WHERE {{
      VALUES ?taxon {{ {qids_str} }}
      OPTIONAL {{ ?taxon wdt:P171 ?taxon_parent. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """


@app.function
def create_taxon_warning_html(
    matches: list, selected_qid: str, is_exact: bool
) -> mo.Html:
    """
    Create an HTML warning with clickable QID links and taxon details.

    Args:
        matches: List of (qid, name, description, parent) tuples
        selected_qid: The QID that was selected
        is_exact: Whether these are exact matches or similar matches
    """
    match_type = "exact matches" if is_exact else "similar taxa"
    intro = (
        f"Ambiguous taxon name. Multiple {match_type} found:"
        if is_exact
        else f"No exact match. Similar taxa found:"
    )

    # Build HTML list of matches
    items = []
    for qid, name, description, parent in matches:
        # Create clickable link
        link = f'<a href="{WIKIDATA_WIKI_PREFIX}{qid}" target="_blank" rel="noopener noreferrer" style="color: {CONFIG["color_hyperlink"]}; text-decoration: none; border-bottom: 1px solid transparent; font-weight: bold;">{qid}</a>'

        # Build details string
        details = []
        if name:
            details.append(f"<em>{name}</em>")
        if description:
            details.append(f"{description}")
        if parent:
            details.append(f"parent: {parent}")

        details_str = " — ".join(details) if details else ""

        # Highlight the selected one
        if qid == selected_qid:
            items.append(
                f"<li>{link} {details_str} <strong>USING THIS ONE BELOW</strong></li>"
            )
        else:
            items.append(f"<li>{link} {details_str}</li>")

    items_html = "".join(items)

    html = f"""
    <div style="line-height: 1.6;">
        {intro}
        <ul style="margin: 0.5em 0; padding-left: 1.5em;">
            {items_html}
        </ul>
        <em>For precision, please use a specific QID directly in the search box.</em>
    </div>
    """

    return mo.Html(html)


@app.function
def resolve_taxon_to_qid(taxon_input: str) -> Tuple[Optional[str], Optional[mo.Html]]:
    """
    Resolve taxon name or QID to a valid QID.

    Returns:
        (qid, warning_html) where warning_html is None if no issues, or mo.Html with clickable links
    """
    taxon_input = taxon_input.strip()

    # Early return if input is already a QID
    if taxon_input.upper().startswith("Q") and taxon_input[1:].isdigit():
        return taxon_input.upper(), None

    # Search for taxon by name
    try:
        query = build_taxon_search_query(taxon_input)
        results = execute_sparql(query)
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            return None, None

        # Extract matches (list comprehension for speed)
        matches = [
            (extract_qid(b["taxon"]["value"]), b["taxon_name"]["value"])
            for b in bindings
            if "taxon" in b
            and "taxon_name" in b
            and "value" in b["taxon"]
            and "value" in b["taxon_name"]
        ]

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

        # Multiple exact matches or similar matches - need to fetch details
        if len(exact_matches) > 1:
            # Get details for exact matches
            qids = [qid for qid, _ in exact_matches]
            details_query = build_taxon_details_query(qids)
            details_results = execute_sparql(details_query)
            details_bindings = details_results.get("results", {}).get("bindings", [])

            # Build a map of QID to details
            details_map = {}
            for b in details_bindings:
                qid = extract_qid(get_binding_value(b, "taxon"))
                details_map[qid] = (
                    get_binding_value(b, "taxonLabel"),
                    get_binding_value(b, "taxonDescription"),
                    get_binding_value(b, "taxon_parentLabel"),
                )

            # Create matches with details
            matches_with_details = [
                (
                    qid,
                    name,
                    details_map.get(qid, ("", "", ""))[1],
                    details_map.get(qid, ("", "", ""))[2],
                )
                for qid, name in exact_matches
            ]

            warning_html = create_taxon_warning_html(
                matches_with_details, exact_matches[0][0], is_exact=True
            )
            return exact_matches[0][0], warning_html

        # No exact match - use first result with warning
        if len(matches) > 1:
            # Get details for similar matches (limit to 5)
            qids = [qid for qid, _ in matches[:5]]
            details_query = build_taxon_details_query(qids)
            details_results = execute_sparql(details_query)
            details_bindings = details_results.get("results", {}).get("bindings", [])

            # Build a map of QID to details
            details_map = {}
            for b in details_bindings:
                qid = extract_qid(get_binding_value(b, "taxon"))
                details_map[qid] = (
                    get_binding_value(b, "taxonLabel"),
                    get_binding_value(b, "taxonDescription"),
                    get_binding_value(b, "taxon_parentLabel"),
                )

            # Create matches with details
            matches_with_details = [
                (
                    qid,
                    name,
                    details_map.get(qid, ("", "", ""))[1],
                    details_map.get(qid, ("", "", ""))[2],
                )
                for qid, name in matches[:5]
            ]

            warning_html = create_taxon_warning_html(
                matches_with_details, matches[0][0], is_exact=False
            )
            return matches[0][0], warning_html

        return matches[0][0], None

    except Exception:
        return None, None


@app.function
def get_binding_value(binding: Dict[str, Any], key: str, default: str = "") -> str:
    return binding.get(key, {}).get("value", default)


@app.function
def create_link(url: str, text: str) -> mo.Html:
    """Create a styled hyperlink."""
    color = CONFIG["color_hyperlink"]
    safe_text = text or url or ""
    safe_url = url or "#"
    return mo.Html(
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
        f'style="color: {color}; text-decoration: none; '
        f'border-bottom: 1px solid transparent; transition: border-color 0.2s;" '
        f"onmouseover=\"this.style.borderColor='{color}'\" "
        f"onmouseout=\"this.style.borderColor='transparent'\">{safe_text}</a>"
    )


@app.function
def create_wikidata_link(qid: str) -> mo.Html:
    """Create a Wikidata link for a QID."""
    return create_link(f"{WIKIDATA_WIKI_PREFIX}{qid}", qid) if qid else mo.Html("—")


@app.function
def pluralize(singular: str, count: int) -> str:
    """Return singular or plural form based on count with special cases."""
    return singular if count == 1 else PLURAL_MAP.get(singular, f"{singular}s")


@app.function
def apply_range_filter(
    df: pl.DataFrame,
    column: str,
    min_val: Optional[float],
    max_val: Optional[float],
    extract_func=None,
) -> pl.DataFrame:
    """
    Generic range filter for DataFrame columns.

    Args:
        df: DataFrame to filter
        column: Column name to filter on
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        extract_func: Optional function to extract value from column (e.g., dt.year())
    """
    if min_val is None or max_val is None or column not in df.columns:
        return df

    col_expr = pl.col(column)
    if extract_func:
        col_expr = extract_func(col_expr)

    return df.filter(
        pl.col(column).is_null() | ((col_expr >= min_val) & (col_expr <= max_val))
    )


@app.function
def apply_year_filter(
    df: pl.DataFrame, year_start: Optional[int], year_end: Optional[int]
) -> pl.DataFrame:
    """Apply year range filter to publication dates."""
    return apply_range_filter(
        df, "pub_date", year_start, year_end, extract_func=lambda col: col.dt.year()
    )


@app.function
def apply_mass_filter(
    df: pl.DataFrame, mass_min: Optional[float], mass_max: Optional[float]
) -> pl.DataFrame:
    """Apply mass range filter."""
    return apply_range_filter(df, "mass", mass_min, mass_max)


@app.function
@lru_cache(maxsize=1024)
def parse_molecular_formula(formula: str) -> tuple:
    """Parse molecular formula and extract atom counts. Returns tuple for caching."""
    if not formula:
        return ()

    # Normalize formula by converting subscripts to regular numbers
    normalized_formula = formula.translate(SUBSCRIPT_MAP)

    # Pattern to match element followed by optional number
    matches = FORMULA_PATTERN.findall(normalized_formula)

    # Return tuple of (element, count) pairs for immutability and caching
    return tuple(
        (element, int(count) if count else 1) for element, count in matches if element
    )


@app.function
def formula_matches_criteria(formula: str, filters: FormulaFilters) -> bool:
    """
    Check if a molecular formula matches the specified criteria.

    Args:
        formula: Molecular formula to check
        filters: FormulaFilters dataclass with all criteria
    """
    if not formula:
        return True  # Keep entries without formula

    # Normalize formula
    normalized_formula = formula.translate(SUBSCRIPT_MAP)

    # Check exact formula match
    if filters.exact_formula and filters.exact_formula.strip():
        normalized_exact = filters.exact_formula.strip().translate(SUBSCRIPT_MAP)
        return normalized_formula == normalized_exact

    # Parse formula (cached)
    atom_tuple = parse_molecular_formula(formula)
    atoms = dict(atom_tuple)

    # Check element ranges
    elements_to_check = [
        ("C", filters.c),
        ("H", filters.h),
        ("N", filters.n),
        ("O", filters.o),
        ("P", filters.p),
        ("S", filters.s),
    ]

    for element, elem_range in elements_to_check:
        if not elem_range.matches(atoms.get(element, 0)):
            return False

    # Check halogens
    halogens = [
        ("F", filters.f_state),
        ("Cl", filters.cl_state),
        ("Br", filters.br_state),
        ("I", filters.i_state),
    ]

    for halogen, state in halogens:
        count = atoms.get(halogen, 0)
        if (state == "required" and count == 0) or (state == "excluded" and count > 0):
            return False

    return True


@app.function
def apply_formula_filter(df: pl.DataFrame, filters: FormulaFilters) -> pl.DataFrame:
    """Apply molecular formula filters to the dataframe."""
    if "mf" not in df.columns or not filters.is_active():
        return df

    # Apply filter using list comprehension (vectorized would be faster but not possible with complex logic)
    mask = [
        formula_matches_criteria(row.get("mf", ""), filters)
        for row in df.iter_rows(named=True)
    ]

    return df.filter(pl.Series(mask))


@app.function
def query_wikidata(
    qid: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
    formula_filters: Optional[FormulaFilters] = None,
) -> pl.DataFrame:
    """
    Query Wikidata for compounds associated with a taxon.

    Args:
        qid: Wikidata QID for taxon
        year_start: Filter start year (inclusive)
        year_end: Filter end year (inclusive)
        mass_min: Minimum mass in Daltons
        mass_max: Maximum mass in Daltons
        formula_filters: Molecular formula filter criteria
    """
    query = build_compounds_query(qid)
    results = execute_sparql(query)
    bindings = results.get("results", {}).get("bindings", [])

    if not bindings:
        return pl.DataFrame()

    # Process results efficiently with list comprehension
    rows = [
        {
            "structure": get_binding_value(b, "compound"),
            "name": get_binding_value(b, "compoundLabel"),
            "inchikey": get_binding_value(b, "compound_inchikey"),
            "smiles": get_binding_value(b, "compound_smiles_iso")
            or get_binding_value(b, "compound_smiles_conn"),
            "taxon_name": get_binding_value(b, "taxon_name"),
            "taxon": get_binding_value(b, "taxon"),
            "ref_title": get_binding_value(b, "ref_title"),
            "ref_doi": (
                doi.split("doi.org/")[-1]
                if (doi := get_binding_value(b, "ref_doi")) and doi.startswith("http")
                else get_binding_value(b, "ref_doi")
            ),
            "reference": get_binding_value(b, "ref_qid"),
            "pub_date": get_binding_value(b, "ref_date", None),
            "mass": float(mass_raw)
            if (mass_raw := get_binding_value(b, "compound_mass", None))
            else None,
            "mf": get_binding_value(b, "compound_formula"),
        }
        for b in bindings
    ]

    df = pl.DataFrame(rows)

    # Optimize date conversion
    if "pub_date" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("pub_date").is_not_null())
            .then(
                pl.col("pub_date")
                .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
                .dt.date()
            )
            .otherwise(None)
            .alias("pub_date")
        )

    # Apply filters (chain for efficiency)
    df = apply_year_filter(df, year_start, year_end)
    df = apply_mass_filter(df, mass_min, mass_max)

    if formula_filters:
        df = apply_formula_filter(df, formula_filters)

    return df.unique(subset=["structure", "taxon", "reference"], keep="first").sort(
        "name"
    )


@app.function
def create_display_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Create a display row for the table with images and links."""
    img_url = create_structure_image_url(row["smiles"])
    struct_qid = extract_qid(row["structure"])
    taxon_qid = extract_qid(row["taxon"])
    ref_qid = extract_qid(row["reference"])
    doi = row["ref_doi"]

    return {
        "2D Depiction": mo.image(src=img_url),
        "Compound": row["name"],
        "Compound SMILES": row["smiles"],
        "Compound InChIKey": row["inchikey"],
        "Taxon": row["taxon_name"],
        "Reference title": row["ref_title"] or "—",
        "Reference DOI": create_link(f"https://doi.org/{doi}", doi)
        if doi
        else mo.Html("—"),
        "Compound QID": create_wikidata_link(struct_qid),
        "Taxon QID": create_wikidata_link(taxon_qid),
        "Reference QID": create_wikidata_link(ref_qid),
    }


@app.function
def prepare_export_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare dataframe for export with cleaned QIDs and selected columns."""
    return df.with_columns(
        [
            pl.col("structure")
            .str.replace(WIKIDATA_ENTITY_PREFIX, "", literal=True)
            .alias("compound_qid"),
            pl.col("taxon")
            .str.replace(WIKIDATA_ENTITY_PREFIX, "", literal=True)
            .alias("taxon_qid"),
            pl.col("reference")
            .str.replace(WIKIDATA_ENTITY_PREFIX, "", literal=True)
            .alias("reference_qid"),
        ]
    ).select(
        [
            pl.col("name").alias("compound_name"),
            pl.col("smiles").alias("compound_smiles"),
            pl.col("inchikey").alias("compound_inchikey"),
            pl.col("mass").alias("compound_mass"),
            pl.col("mf").alias("molecular_formula"),
            "taxon_name",
            pl.col("ref_title").alias("reference_title"),
            pl.col("ref_doi").alias("reference_doi"),
            pl.col("pub_date").alias("reference_date"),
            "compound_qid",
            "taxon_qid",
            "reference_qid",
        ]
    )


@app.function
def create_export_metadata(
    df: pl.DataFrame, taxon_input: str, qid: str, filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create metadata for exported data following FAIR principles.

    Returns machine-readable metadata with provenance, access info, and citations.
    """
    return {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": f"LOTUS Data for {taxon_input}",
        "description": f"Chemical compounds found in taxon {taxon_input} (Wikidata QID: {qid})",
        "version": "0.0.1",
        "dateCreated": datetime.now().isoformat(),
        "license": "https://creativecommons.org/publicdomain/zero/1.0/",
        "creator": {
            "@type": "SoftwareApplication",
            "name": "LOTUS Wikidata Explorer",
            "version": "0.0.1",
            "url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
            "license": "https://www.gnu.org/licenses/agpl-3.0.html",
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
                "url": "https://www.wikidata.org/",
            },
        ],
        "citation": [
            {
                "@type": "ScholarlyArticle",
                "name": "The LOTUS initiative for open knowledge management in natural products research",
                "identifier": "https://doi.org/10.7554/eLife.70780",
            }
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
        ],
        "numberOfRecords": len(df),
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
            "taxon_qid": qid,
            "filters": filters,
        },
        "sparql_endpoint": "https://query.wikidata.org/sparql",
    }


@app.function
def create_citation_text(taxon_input: str) -> str:
    """Generate citation text for the exported data."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"""
## How to Cite This Data

### Dataset Citation
LOTUS Initiative via Wikidata. ({datetime.now().year}). Data for {taxon_input}. 
Retrieved from LOTUS Wikidata Explorer on {current_date}.
Available under CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

### Original LOTUS Initiative
Rutz A, Sorokina M, Galgonek J, et al. (2022). The LOTUS initiative for open 
knowledge management in natural products research. eLife 11:e70780.
https://doi.org/10.7554/eLife.70780

### Software
LOTUS Wikidata Explorer v0.0.1. Available at: 
https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py
Licensed under AGPL-3.0.

### Data Sources
- Wikidata (https://www.wikidata.org) - CC0 1.0
- LOTUS (https://www.wikidata.org/wiki/Q104225190) - CC0 1.0
"""


@app.cell
def _():
    mo.md("""
    # 🌿 LOTUS Wikidata Explorer

    Explore chemical compounds from [LOTUS](https://doi.org/10.7554/eLife.70780) and 
    [Wikidata](https://www.wikidata.org/) for any taxon.

    Enter a taxon name to discover chemical compounds found in organisms of that taxonomic group.

    💡 **New to this tool?** Open the "Help & Documentation" section below for a quick start guide.
    """)
    return


@app.cell
def _(
    state_br_state,
    state_c_max,
    state_c_min,
    state_cl_state,
    state_exact_formula,
    state_f_state,
    state_formula_filter,
    state_h_max,
    state_h_min,
    state_i_state,
    state_mass_filter,
    state_mass_max,
    state_mass_min,
    state_n_max,
    state_n_min,
    state_o_max,
    state_o_min,
    state_p_max,
    state_p_min,
    state_s_max,
    state_s_min,
    state_taxon,
    state_year_end,
    state_year_filter,
    state_year_start,
):
    ## MASS FILTERS
    mass_filter = mo.ui.checkbox(label="⚖ Filter by mass", value=state_mass_filter)

    mass_min = mo.ui.number(
        value=state_mass_min,
        start=0,
        stop=10000,
        step=10,
        label="Min mass (Da)",
        full_width=True,
    )

    mass_max = mo.ui.number(
        value=state_mass_max,
        start=0,
        stop=10000,
        step=10,
        label="Max mass (Da)",
        full_width=True,
    )

    ## FORMULA FILTERS
    formula_filter = mo.ui.checkbox(
        label="⚛ Filter by molecular formula", value=state_formula_filter
    )

    exact_formula = mo.ui.text(
        value=state_exact_formula,
        label="Exact formula (e.g., C15H10O5)",
        placeholder="Leave empty to use element ranges",
        full_width=True,
    )

    c_min = mo.ui.number(
        value=state_c_min, start=0, stop=100, label="C min", full_width=True
    )
    c_max = mo.ui.number(
        value=state_c_max, start=0, stop=100, label="C max", full_width=True
    )
    h_min = mo.ui.number(
        value=state_h_min, start=0, stop=200, label="H min", full_width=True
    )
    h_max = mo.ui.number(
        value=state_h_max, start=0, stop=200, label="H max", full_width=True
    )
    n_min = mo.ui.number(
        value=state_n_min, start=0, stop=50, label="N min", full_width=True
    )
    n_max = mo.ui.number(
        value=state_n_max, start=0, stop=50, label="N max", full_width=True
    )
    o_min = mo.ui.number(
        value=state_o_min, start=0, stop=50, label="O min", full_width=True
    )
    o_max = mo.ui.number(
        value=state_o_max, start=0, stop=50, label="O max", full_width=True
    )
    p_min = mo.ui.number(
        value=state_p_min, start=0, stop=20, label="P min", full_width=True
    )
    p_max = mo.ui.number(
        value=state_p_max, start=0, stop=20, label="P max", full_width=True
    )
    s_min = mo.ui.number(
        value=state_s_min, start=0, stop=20, label="S min", full_width=True
    )
    s_max = mo.ui.number(
        value=state_s_max, start=0, stop=20, label="S max", full_width=True
    )

    # Halogen selectors (allowed/required/excluded)
    halogen_options = ["allowed", "required", "excluded"]
    f_state = mo.ui.dropdown(
        options=halogen_options, value=state_f_state, label="F", full_width=True
    )
    cl_state = mo.ui.dropdown(
        options=halogen_options, value=state_cl_state, label="Cl", full_width=True
    )
    br_state = mo.ui.dropdown(
        options=halogen_options, value=state_br_state, label="Br", full_width=True
    )
    i_state = mo.ui.dropdown(
        options=halogen_options, value=state_i_state, label="I", full_width=True
    )

    ## DATE FILTERS
    current_year = datetime.now().year

    taxon_input = mo.ui.text(
        value=state_taxon,
        label="🔬 Taxon name or QID",
        placeholder="e.g., Swertia chirayita, Anabaena, Q157115, ...",
        full_width=True,
    )

    year_filter = mo.ui.checkbox(
        label="⏱ Filter by publication year", value=state_year_filter
    )

    year_start = mo.ui.number(
        value=state_year_start,
        start=1700,
        stop=current_year,
        label="Start year",
        full_width=True,
    )

    year_end = mo.ui.number(
        value=state_year_end,
        start=1700,
        stop=current_year,
        label="End year",
        full_width=True,
    )

    run_button = mo.ui.run_button(label="🔍 Search Wikidata")
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
        taxon_input,
        year_end,
        year_filter,
        year_start,
    )


@app.cell
def _(
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
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    filters_ui = [
        mo.md("## Search Parameters"),
        taxon_input,
        mo.hstack([mass_filter], justify="start"),
        mo.hstack([mass_min, mass_max], gap=2, widths="equal")
        if mass_filter.value
        else mo.Html(""),
        mo.hstack([formula_filter], justify="start"),
        mo.hstack([year_filter], justify="start"),
        mo.hstack([year_start, year_end], gap=2, widths="equal")
        if year_filter.value
        else mo.Html(""),
    ]

    if formula_filter.value:
        filters_ui.extend(
            [
                exact_formula,
                mo.md("**Element ranges** (leave empty to ignore)"),
                mo.hstack([c_min, c_max], gap=2, widths="equal"),
                mo.hstack([h_min, h_max], gap=2, widths="equal"),
                mo.hstack([n_min, n_max], gap=2, widths="equal"),
                mo.hstack([o_min, o_max], gap=2, widths="equal"),
                mo.hstack([p_min, p_max], gap=2, widths="equal"),
                mo.hstack([s_min, s_max], gap=2, widths="equal"),
                mo.md("**Halogens** (allowed / required / excluded)"),
                mo.hstack(
                    [f_state, cl_state, br_state, i_state], gap=2, widths="equal"
                ),
            ]
        )

    filters_ui.append(run_button)
    mo.vstack(filters_ui)
    return


@app.cell
def _(
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
    state_auto_run,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    # Auto-run if URL parameters were detected, or if run button was clicked
    if not run_button.value and not state_auto_run:
        results_df = None
        qid = None
        taxon_warning = None
    else:
        taxon_input_str = taxon_input.value.strip()
        start_time = time.time()
        with mo.status.spinner(title=f"🔎 Querying Wikidata for {taxon_input_str}..."):
            qid, taxon_warning = resolve_taxon_to_qid(taxon_input_str)
            if not qid:
                mo.stop(
                    True,
                    mo.callout(
                        mo.md(
                            f"**Taxon not found:** Could not find '{taxon_input_str}' in Wikidata. Please check the spelling or try a different taxonomic name."
                        ),
                        kind="warn",
                    ),
                )

            try:
                y_start = year_start.value if year_filter.value else None
                y_end = year_end.value if year_filter.value else None
                m_min = mass_min.value if mass_filter.value else None
                m_max = mass_max.value if mass_filter.value else None

                # Build formula filters using dataclass
                formula_filt = None
                if formula_filter.value:
                    formula_filt = FormulaFilters(
                        exact_formula=exact_formula.value
                        if exact_formula.value.strip()
                        else None,
                        c=ElementRange(c_min.value, c_max.value),
                        h=ElementRange(h_min.value, h_max.value),
                        n=ElementRange(n_min.value, n_max.value),
                        o=ElementRange(o_min.value, o_max.value),
                        p=ElementRange(p_min.value, p_max.value),
                        s=ElementRange(s_min.value, s_max.value),
                        f_state=f_state.value,
                        cl_state=cl_state.value,
                        br_state=br_state.value,
                        i_state=i_state.value,
                    )

                results_df = query_wikidata(
                    qid, y_start, y_end, m_min, m_max, formula_filt
                )
            except Exception as e:
                mo.stop(
                    True, mo.callout(mo.md(f"**Query Error:** {str(e)}"), kind="danger")
                )
        elapsed = round(time.time() - start_time, 2)
        mo.md(f"⏱️ Query completed in **{elapsed}s**.")
    return qid, results_df, taxon_warning


@app.cell
def _(qid, results_df, run_button, state_auto_run, taxon_input, taxon_warning):
    # Display summary if either button was clicked or auto-run from URL
    if (not run_button.value and not state_auto_run) or results_df is None:
        summary_display = mo.Html("")
    elif len(results_df) == 0:
        # Show no compounds message, and taxon warning if present
        parts = []
        if taxon_warning:
            parts.append(mo.callout(taxon_warning, kind="warn"))
        parts.append(
            mo.callout(
                mo.md(
                    f"No natural products found for **{taxon_input.value}** ({create_wikidata_link(qid)}) with the current filters."
                ),
                kind="warn",
            )
        )
        summary_display = mo.vstack(parts) if len(parts) > 1 else parts[0]
    else:
        n_compounds = results_df.n_unique(subset=["structure"])
        n_taxa = results_df.n_unique(subset=["taxon"])
        n_refs = results_df.n_unique(subset=["reference"])
        n_entries = len(results_df)

        summary_parts = [
            mo.md(
                f"## Results Summary\n\nFound data for **{taxon_input.value}** {create_wikidata_link(qid)}"
            ),
        ]

        if taxon_warning:
            summary_parts.append(mo.callout(taxon_warning, kind="warn"))

        summary_parts.append(
            mo.hstack(
                [
                    mo.stat(
                        value=str(n_compounds),
                        label=f"🧪 {pluralize('Compound', n_compounds)}",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(n_taxa),
                        label=f"🌱 {pluralize('Taxon', n_taxa)}",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(n_refs),
                        label=f"📚 {pluralize('Reference', n_refs)}",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(n_entries),
                        label=f"📝 {pluralize('Entry', n_entries)}",
                        bordered=True,
                    ),
                ],
                gap=2,
                justify="start",
                wrap=True,
            )
        )

        summary_display = mo.vstack(summary_parts)

    summary_display
    return


@app.cell
def _(
    formula_filter,
    mass_filter,
    mass_max,
    mass_min,
    qid,
    results_df,
    run_button,
    state_auto_run,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    # Display table if either button was clicked or auto-run from URL
    if (not run_button.value and not state_auto_run) or results_df is None:
        table_output = None
    elif len(results_df) == 0:
        table_output = mo.callout(
            mo.md("No compounds match your search criteria."), kind="neutral"
        )
    else:
        # Create display table
        display_data = [
            create_display_row(row) for row in results_df.iter_rows(named=True)
        ]

        display_table = mo.ui.table(
            display_data,
            selection=None,
            page_size=CONFIG["page_size_default"],
            show_column_summaries=False,
        )

        # Create export table (preserves pub_date as actual date)
        export_df = prepare_export_dataframe(results_df)
        export_data = export_df.to_dicts()
        export_table = mo.ui.table(
            export_data,
            selection=None,
            page_size=CONFIG["page_size_export"],
            show_column_summaries=False,
        )

        # Create export files
        csv_data = export_df.write_csv()
        json_data = export_df.write_json()

        # Get active filters for metadata
        active_filters = {
            "mass_filter": mass_filter.value if "mass_filter" in dir() else False,
            "mass_min": mass_min.value
            if "mass_min" in dir() and mass_filter.value
            else None,
            "mass_max": mass_max.value
            if "mass_max" in dir() and mass_filter.value
            else None,
            "year_filter": year_filter.value if "year_filter" in dir() else False,
            "year_start": year_start.value
            if "year_start" in dir() and year_filter.value
            else None,
            "year_end": year_end.value
            if "year_end" in dir() and year_filter.value
            else None,
            "formula_filter": formula_filter.value
            if "formula_filter" in dir()
            else False,
        }

        metadata = create_export_metadata(
            export_df, taxon_input.value, qid, active_filters
        )
        import json

        metadata_json = json.dumps(metadata, indent=2)

        citation_text = create_citation_text(taxon_input.value)

        # Download buttons
        download_buttons = mo.hstack(
            [
                mo.download(
                    data=csv_data,
                    filename=f"lotus_data_{taxon_input.value.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    label="📥 CSV",
                    mimetype="text/csv",
                ),
                mo.download(
                    data=json_data,
                    filename=f"lotus_data_{taxon_input.value.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                    label="📥 JSON",
                    mimetype="application/json",
                ),
                mo.download(
                    data=metadata_json,
                    filename=f"lotus_metadata_{taxon_input.value.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                    label="📋 Metadata",
                    mimetype="application/json",
                ),
            ],
            gap=2,
            wrap=True,
        )

        table_output = mo.vstack(
            [
                mo.md("### Data Tables"),
                download_buttons,
                mo.ui.tabs(
                    {
                        "🖼️  Display": display_table,
                        "📥 Export": export_table,
                        "📖 Citation": mo.md(citation_text),
                        "🏷️  Metadata": mo.md(f"```json\n{metadata_json}\n```"),
                    }
                ),
            ]
        )

    table_output
    return


@app.cell
def _():
    mo.accordion(
        {
            "🔗 URL Query API": mo.md("""
            You can query this notebook via URL parameters! When running locally or accessing the published version, add query parameters to automatically execute searches.

            ### Available Parameters

            - `taxon` - Taxon name or QID (required)
            - `mass_min`, `mass_max` - Mass range in Daltons
            - `year_start`, `year_end` - Publication year range
            - `exact_formula` - Exact molecular formula (e.g., C15H10O5)
            - `c_min`, `c_max` - Carbon count range
            - `h_min`, `h_max` - Hydrogen count range
            - `n_min`, `n_max` - Nitrogen count range
            - `o_min`, `o_max` - Oxygen count range
            - `p_min`, `p_max` - Phosphorus count range
            - `s_min`, `s_max` - Sulfur count range
            - `f_state`, `cl_state`, `br_state`, `i_state` - Halogen states (allowed/required/excluded)

            ### Examples

            #### Search by taxon name with mass filter

            ```text
            ?taxon=Swertia&mass_min=200&mass_max=600
            ```

            #### Search by QID with year and carbon range

            ```text
            ?taxon=Q157115&year_start=2000&c_min=15&c_max=25
            ```

            #### Search excluding fluorine and requiring chlorine

            ```text
            ?taxon=Artemisia&f_state=excluded&cl_state=required
            ```

            **Tip:** Copy the query parameters above and append them to your notebook URL.
            """),
            "❓ Help & Documentation": mo.md("""
            ### Quick Start Guide

            1. **Enter a taxon name** (e.g., "Artemisia annua") or Wikidata QID (e.g., "Q157115")
            2. **Optional:** Apply filters for mass, publication year, or molecular formula
            3. **Click "🔍 Search Wikidata"** to retrieve data
            4. **Download** your results in CSV, JSON, or with full metadata

            ### Features

            #### Taxon Search
            - Search by scientific name (case-insensitive)
            - Search directly by Wikidata QID for precision
            - Handles ambiguous names with helpful suggestions

            #### Filtering Options

            **Mass Filter** ⚖  
            Filter compounds by molecular mass (in Daltons)

            **Molecular Formula Filter** ⚛  
            - Search by exact formula (e.g., C15H10O5)
            - Set element ranges (C, H, N, O, P, S)
            - Control halogen presence (F, Cl, Br, I):
              - *Allowed*: Can be present or absent
              - *Required*: Must be present
              - *Excluded*: Must not be present

            **Publication Year Filter** ⏱  
            Filter by the year references were published

            #### Data Export

            - **CSV**: Spreadsheet-compatible format
            - **JSON**: Machine-readable structured data
            - **Metadata**: Schema.org-compliant metadata with provenance
            - **Citation**: Proper citations for your publications
            """),
        }
    )
    return


@app.cell
def _():
    url_params = mo.query_params()

    # Display URL query info if parameters are present
    # QueryParams has keys() method, and we can access values with get()
    if url_params and hasattr(url_params, "keys") and len(list(url_params.keys())) > 0:
        param_list = [f"**{k}**: {url_params.get(k)}" for k in url_params.keys()]
        mo.callout(
            mo.md(f"""
            ### 🔗 URL Query Detected

            {chr(10).join(param_list)}

            The search will auto-execute with these parameters.
            """),
            kind="info",
        )
    return (url_params,)


@app.cell
def _(url_params):
    # Parse URL parameters and set defaults for the query
    def get_url_param(key: str, default=None, param_type=str):
        """Get URL parameter with type conversion."""
        value = url_params.get(key)
        if value is None:
            return default
        try:
            if param_type == bool:
                return value.lower() in ("true", "1", "yes")
            elif param_type == int:
                return int(value)
            elif param_type == float:
                return float(value)
            else:
                return str(value)
        except (ValueError, AttributeError):
            return default

    def get_element_range_params(
        prefix: str, default_max: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Helper to get min/max for an element range."""
        return (
            get_url_param(f"{prefix}_min", None, int),
            get_url_param(f"{prefix}_max", default_max, int),
        )

    # Extract all parameters from URL
    url_taxon = get_url_param("taxon")
    url_mass_filter = (
        get_url_param("mass_min") is not None or get_url_param("mass_max") is not None
    )
    url_mass_min = get_url_param("mass_min", 0, float)
    url_mass_max = get_url_param("mass_max", 2000, float)
    url_year_filter = (
        get_url_param("year_start") is not None or get_url_param("year_end") is not None
    )
    url_year_start = get_url_param("year_start", 1900, int)
    url_year_end = get_url_param("year_end", 2025, int)

    # Check if formula filter is active
    url_formula_filter = any(
        [
            get_url_param("exact_formula"),
            *[
                get_url_param(f"{e}_min") or get_url_param(f"{e}_max")
                for e in ["c", "h", "n", "o", "p", "s"]
            ],
            get_url_param("f_state", "allowed") != "allowed",
            get_url_param("cl_state", "allowed") != "allowed",
            get_url_param("br_state", "allowed") != "allowed",
            get_url_param("i_state", "allowed") != "allowed",
        ]
    )

    url_exact_formula = get_url_param("exact_formula", "")

    # Use helper for element ranges
    url_c_min, url_c_max = get_element_range_params("c", 100)
    url_h_min, url_h_max = get_element_range_params("h", 200)
    url_n_min, url_n_max = get_element_range_params("n", 50)
    url_o_min, url_o_max = get_element_range_params("o", 50)
    url_p_min, url_p_max = get_element_range_params("p", 20)
    url_s_min, url_s_max = get_element_range_params("s", 20)

    # Halogen states
    url_f_state = get_url_param("f_state", "allowed")
    url_cl_state = get_url_param("cl_state", "allowed")
    url_br_state = get_url_param("br_state", "allowed")
    url_i_state = get_url_param("i_state", "allowed")

    # Auto-trigger search if taxon is in URL
    url_auto_search = url_taxon is not None
    return (
        url_auto_search,
        url_br_state,
        url_c_max,
        url_c_min,
        url_cl_state,
        url_exact_formula,
        url_f_state,
        url_formula_filter,
        url_h_max,
        url_h_min,
        url_i_state,
        url_mass_filter,
        url_mass_max,
        url_mass_min,
        url_n_max,
        url_n_min,
        url_o_max,
        url_o_min,
        url_p_max,
        url_p_min,
        url_s_max,
        url_s_min,
        url_taxon,
        url_year_end,
        url_year_filter,
        url_year_start,
    )


@app.cell
def _(
    url_auto_search,
    url_br_state,
    url_c_max,
    url_c_min,
    url_cl_state,
    url_exact_formula,
    url_f_state,
    url_formula_filter,
    url_h_max,
    url_h_min,
    url_i_state,
    url_mass_filter,
    url_mass_max,
    url_mass_min,
    url_n_max,
    url_n_min,
    url_o_max,
    url_o_min,
    url_p_max,
    url_p_min,
    url_s_max,
    url_s_min,
    url_taxon,
    url_year_end,
    url_year_filter,
    url_year_start,
):
    # Create state variables that will be used to populate the UI
    # These are only set when URL parameters are present
    if url_auto_search:
        # Taxon state
        state_taxon = url_taxon or "Gentiana lutea"

        # Mass filter state
        state_mass_filter = url_mass_filter
        state_mass_min = url_mass_min if url_mass_filter else 0
        state_mass_max = url_mass_max if url_mass_filter else 2000

        # Year filter state
        state_year_filter = url_year_filter
        state_year_start = url_year_start if url_year_filter else 1900
        state_year_end = url_year_end if url_year_filter else 2025

        # Formula filter state
        state_formula_filter = url_formula_filter
        state_exact_formula = url_exact_formula if url_formula_filter else ""
        state_c_min = url_c_min if url_formula_filter else None
        state_c_max = url_c_max if url_formula_filter else 100
        state_h_min = url_h_min if url_formula_filter else None
        state_h_max = url_h_max if url_formula_filter else 200
        state_n_min = url_n_min if url_formula_filter else None
        state_n_max = url_n_max if url_formula_filter else 50
        state_o_min = url_o_min if url_formula_filter else None
        state_o_max = url_o_max if url_formula_filter else 50
        state_p_min = url_p_min if url_formula_filter else None
        state_p_max = url_p_max if url_formula_filter else 20
        state_s_min = url_s_min if url_formula_filter else None
        state_s_max = url_s_max if url_formula_filter else 20
        state_f_state = url_f_state if url_formula_filter else "allowed"
        state_cl_state = url_cl_state if url_formula_filter else "allowed"
        state_br_state = url_br_state if url_formula_filter else "allowed"
        state_i_state = url_i_state if url_formula_filter else "allowed"

        state_auto_run = True

        mo.md(f"**Auto-executing search for:** {url_taxon}")
    else:
        # Default states when no URL parameters
        state_taxon = "Gentiana lutea"
        state_mass_filter = False
        state_mass_min = 0
        state_mass_max = 2000
        state_year_filter = False
        state_year_start = 1900
        state_year_end = 2025
        state_formula_filter = False
        state_exact_formula = ""
        state_c_min = None
        state_c_max = 100
        state_h_min = None
        state_h_max = 200
        state_n_min = None
        state_n_max = 50
        state_o_min = None
        state_o_max = 50
        state_p_min = None
        state_p_max = 20
        state_s_min = None
        state_s_max = 20
        state_f_state = "allowed"
        state_cl_state = "allowed"
        state_br_state = "allowed"
        state_i_state = "allowed"
        state_auto_run = False
    return (
        state_auto_run,
        state_br_state,
        state_c_max,
        state_c_min,
        state_cl_state,
        state_exact_formula,
        state_f_state,
        state_formula_filter,
        state_h_max,
        state_h_min,
        state_i_state,
        state_mass_filter,
        state_mass_max,
        state_mass_min,
        state_n_max,
        state_n_min,
        state_o_max,
        state_o_min,
        state_p_max,
        state_p_min,
        state_s_max,
        state_s_min,
        state_taxon,
        state_year_end,
        state_year_filter,
        state_year_start,
    )


@app.cell
def _():
    mo.md(
        """
    ---
    **Data:** <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> & <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a>  |  
    **Code:** <a href="https://github.com/cdk/depict" style="color:#339966;">CDK Depict</a> & 
    <a href="https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py" style="color:#339966;">lotus_wikidata_explorer.py</a>  |  
    **License:** <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#006699;">CC0 1.0</a> for data & <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#006699;">AGPL-3.0</a> for code
    """
    )
    return


if __name__ == "__main__":
    app.run()
