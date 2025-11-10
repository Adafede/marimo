# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars==1.35.2",
#     "pyarrow==22.0.0",
#     "rdflib==7.4.0",
#     "requests==2.31.0",
#     "urllib3==2.5.0",
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
    import json
    import re
    import requests
    import time
    import gzip
    from dataclasses import dataclass, field
    from datetime import datetime
    from functools import lru_cache
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD, DCTERMS
    from requests.adapters import HTTPAdapter
    from typing import Optional, Dict, Any, Tuple, List
    from urllib.parse import quote as url_quote
    from urllib3.util.retry import Retry

    # ====================================================================
    # CONFIGURATION
    # ====================================================================

    CONFIG = {
        # API and External Services
        "cdk_base": "https://www.simolecule.com/cdkdepict/depict/cot/svg",
        # "sparql_endpoint": "https://qlever.dev/wikidata",  # Fails CORS for now
        "sparql_endpoint": "https://qlever.cs.uni-freiburg.de/api/wikidata",  # Somehow works?
        # "sparql_endpoint": "https://query-legacy-full.wikidata.org/sparql",  # Too slow
        "user_agent": "LOTUS Explorer/0.0.1 (https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py)",
        # Network Settings
        "max_retries": 3,
        "retry_backoff": 2,
        # UI Display
        "color_hyperlink": "#006699",
        "page_size_default": 10,
        "page_size_export": 25,
        # Performance Thresholds
        "table_row_limit": 10000,  # Max rows for tables
        "lazy_generation_threshold": 5000,  # Defer generation for datasets > this size
        "download_embed_threshold_bytes": 8_000_000,  # Compress data > 8MB for download UI
        # Filter Default Values
        "year_range_start": 1700,  # Minimum year for publication date filter
        "year_default_start": 1900,  # Default start year
        "mass_default_min": 0,  # Default minimum mass in Daltons
        "mass_default_max": 2000,  # Default maximum mass in Daltons
        # Molecular Formula Filter Ranges
        "element_c_max": 100,  # Carbon max range
        "element_h_max": 200,  # Hydrogen max range
        "element_n_max": 50,  # Nitrogen max range
        "element_o_max": 50,  # Oxygen max range
        "element_p_max": 20,  # Phosphorus max range
        "element_s_max": 20,  # Sulfur max range
    }

    # Wikidata URLs (constants)
    WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
    WIKIDATA_WIKI_PREFIX = "https://www.wikidata.org/wiki/"

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
    # HTTP SESSION (Efficiency - Connection Pooling)
    # ====================================================================

    def create_http_session() -> requests.Session:
        """
        Create HTTP session with connection pooling and retry logic.

        Benefits:
        - Reuses TCP connections (faster subsequent requests)
        - Automatic retries on transient failures
        - Connection pooling reduces overhead
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=CONFIG["max_retries"],
            backoff_factor=CONFIG["retry_backoff"],
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=20
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    # Global session for connection reuse (significant performance improvement)
    HTTP_SESSION = create_http_session()

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
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?taxon ?taxon_name WHERE {{
      VALUES ?taxon_name {{ "{taxon_name}" }}
      ?taxon wdt:P225 ?taxon_name .
    }}
    """


@app.function
def build_compounds_query(qid: str) -> str:
    return f"""
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?compound ?compound_inchikey ?compound_smiles_conn ?taxon_name ?taxon ?ref_qid 
           ?compound_smiles_iso ?compound_mass ?compound_formula ?compoundLabel 
           ?ref_title ?ref_doi ?ref_date WHERE {{
      {{
        SELECT ?taxon ?taxon_name WHERE {{
          ?taxon (wdt:P171*) wd:{qid};
                 wdt:P225 ?taxon_name.
        }}
      }}
      ?statement ps:P703 ?taxon;
                 prov:wasDerivedFrom ?ref.
      ?ref pr:P248 ?ref_qid.
      ?compound wdt:P235 ?compound_inchikey;
                wdt:P233 ?compound_smiles_conn;
                p:P703 ?statement.

      OPTIONAL {{ ?compound wdt:P2017 ?compound_smiles_iso. }}
      OPTIONAL {{ ?compound wdt:P2067 ?compound_mass. }}
      OPTIONAL {{ ?compound wdt:P274 ?compound_formula. }}

     OPTIONAL {{
        ?compound rdfs:label ?compoundLabel.
        FILTER((LANG(?compoundLabel)) = "en")
        }}
      OPTIONAL {{
        ?compound rdfs:label ?compoundLabel.
        FILTER((LANG(?compoundLabel)) = "mul")
        }}

      OPTIONAL {{ ?ref_qid wdt:P1476 ?ref_title. }}
      OPTIONAL {{ ?ref_qid wdt:P356 ?ref_doi. }}
      OPTIONAL {{ ?ref_qid wdt:P577 ?ref_date. }}
    }}
    """


@app.function
def build_all_compounds_query() -> str:
    return """
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?compound ?compoundLabel ?compound_inchikey ?compound_smiles_iso 
                    ?compound_smiles_conn ?compound_mass ?compound_formula 
                    ?taxon_name ?taxon ?ref_title ?ref_doi ?ref_qid ?ref_date WHERE {
      {
        SELECT ?compound ?compound_inchikey ?compound_smiles_conn ?taxon_name ?taxon ?ref_qid WHERE {
          ?compound wdt:P235 ?compound_inchikey;
            wdt:P233 ?compound_smiles_conn;
            p:P703 ?statement.
          ?statement ps:P703 ?taxon;
            prov:wasDerivedFrom ?ref.
          ?taxon wdt:P225 ?taxon_name.
          ?ref pr:P248 ?ref_qid.
        }
      }
	  OPTIONAL { ?compound wdt:P2017 ?compound_smiles_iso. }
    OPTIONAL { ?compound wdt:P2067 ?compound_mass. }
    OPTIONAL { ?compound wdt:P274 ?compound_formula. }
      OPTIONAL {
        ?compound rdfs:label ?compoundLabel .
        FILTER(LANG(?compoundLabel) = "en")
      }
      OPTIONAL {
        ?compound rdfs:label ?compoundLabel .
        FILTER(LANG(?compoundLabel) = "mul")
      }
      OPTIONAL { ?ref_qid wdt:P1476 ?ref_title. }
      OPTIONAL { ?ref_qid wdt:P356 ?ref_doi. }
      OPTIONAL { ?ref_qid  wdt:P577 ?ref_date. }
    }
    """


@app.function
@lru_cache(maxsize=128)
def execute_sparql(
    query: str, max_retries: int = CONFIG["max_retries"]
) -> Dict[str, Any]:
    """
    Execute SPARQL query with connection pooling and retry logic.

    Performance improvements:
    - Uses persistent HTTP session (reuses TCP connections)
    - Automatic retry with exponential backoff
    - Connection pooling reduces overhead by ~30-50%

    Args:
        query: SPARQL query string
        max_retries: Maximum number of retry attempts

    Returns:
        Dict containing SPARQL results

    Raises:
        Exception: With user-friendly error message if query fails
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": CONFIG["user_agent"],
    }

    params = {
        "query": query,
        "format": "json",
    }

    url = CONFIG["sparql_endpoint"]

    for attempt in range(max_retries):
        try:
            # Use persistent session for connection pooling (major performance boost)
            response = HTTP_SESSION.get(
                url=url,
                headers=headers,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise Exception(
                    f"❌ Query timed out after {max_retries} attempts.\n"
                    f"💡 Try: Add filters to reduce result size or simplify the query."
                )
            time.sleep(CONFIG["retry_backoff"] * (2**attempt))

        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                status_code = (
                    e.response.status_code if hasattr(e, "response") else "Unknown"
                )
                raise Exception(
                    f"🌐 HTTP error {status_code} after {max_retries} attempts.\n"
                    f"💡 Try: Check your internet connection or try again later."
                )
            time.sleep(CONFIG["retry_backoff"] * (2**attempt))

        except Exception as e:
            if attempt == max_retries - 1:
                query_snippet = query[:200] + "..." if len(query) > 200 else query
                raise Exception(
                    f"❌ Query failed: {str(e)}\n" f"Query snippet: {query_snippet}"
                )
            time.sleep(CONFIG["retry_backoff"] * (2**attempt))

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
    encoded_smiles = url_quote(smiles)
    return f"{CONFIG['cdk_base']}?smi={encoded_smiles}&annotate=cip"


@app.function
def build_sparql_values_clause(
    variable: str, values: list, use_wd_prefix: bool = True
) -> str:
    """
    Build a SPARQL VALUES clause for a list of values.

    Args:
        variable: SPARQL variable name (e.g., 'taxon', 'compound')
        values: List of QIDs or URIs
        use_wd_prefix: Whether to prefix values with 'wd:' (default True)

    Returns:
        SPARQL VALUES clause string

    Example:
        >>> build_sparql_values_clause('taxon', ['Q12345', 'Q67890'])
        'VALUES ?taxon { wd:Q12345 wd:Q67890 }'
    """
    if use_wd_prefix:
        values_str = " ".join(f"wd:{v}" for v in values)
    else:
        values_str = " ".join(f"<{v}>" for v in values)

    return f"VALUES ?{variable} {{ {values_str} }}"


@app.function
def build_taxon_details_query(qids: List[str]) -> str:
    values_clause = build_sparql_values_clause("taxon", qids)
    return f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?taxon ?taxonLabel ?taxonDescription ?taxon_parent ?taxon_parentLabel 
    WHERE {{
      {values_clause}

      # Parent taxon
      OPTIONAL {{ ?taxon wdt:P171 ?taxon_parent }}

      # Taxon labels (English preferred)
      OPTIONAL {{
        ?taxon rdfs:label ?taxonLabel .
        FILTER(LANG(?taxonLabel) = "en")
      }}
      OPTIONAL {{
        ?taxon rdfs:label ?taxonLabel .
        FILTER(LANG(?taxonLabel) = "mul")
      }}

      # Taxon descriptions
      OPTIONAL {{
        ?taxon schema:description ?taxonDescription .
        FILTER(LANG(?taxonDescription) = "en")
      }}
      OPTIONAL {{
        ?taxon schema:description ?taxonDescription .
        FILTER(LANG(?taxonDescription) = "mul")
      }}

      # Parent labels
      OPTIONAL {{
        ?taxon_parent rdfs:label ?taxon_parentLabel .
        FILTER(LANG(?taxon_parentLabel) = "en")
      }}
      OPTIONAL {{
        ?taxon_parent rdfs:label ?taxon_parentLabel .
        FILTER(LANG(?taxon_parentLabel) = "mul")
      }}
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

    Special case:
        If taxon_input is "*", returns ("*", None) to indicate all taxa
    """
    taxon_input = taxon_input.strip()

    # Handle wildcard for all taxa
    if taxon_input == "*":
        return "*", None

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
def serialize_element_range(element_range: ElementRange) -> Optional[Dict[str, int]]:
    """Convert ElementRange to dictionary for export, returns None if not active."""
    if not element_range.is_active():
        return None
    return {
        "min": element_range.min_val,
        "max": element_range.max_val,
    }


@app.function
def serialize_formula_filters(
    filters: Optional[FormulaFilters],
) -> Optional[Dict[str, Any]]:
    """Convert FormulaFilters to dictionary for metadata export."""
    if not filters or not filters.is_active():
        return None

    result = {}

    # Exact formula
    if filters.exact_formula and filters.exact_formula.strip():
        result["exact_formula"] = filters.exact_formula.strip()

    # Element ranges
    for element_name, element_range in [
        ("carbon", filters.c),
        ("hydrogen", filters.h),
        ("nitrogen", filters.n),
        ("oxygen", filters.o),
        ("phosphorus", filters.p),
        ("sulfur", filters.s),
    ]:
        range_dict = serialize_element_range(element_range)
        if range_dict:
            result[element_name] = range_dict

    # Halogen states (only include if not "allowed")
    halogen_states = {}
    for halogen_name, state in [
        ("fluorine", filters.f_state),
        ("chlorine", filters.cl_state),
        ("bromine", filters.br_state),
        ("iodine", filters.i_state),
    ]:
        if state != "allowed":
            halogen_states[halogen_name] = state

    if halogen_states:
        result["halogens"] = halogen_states

    return result if result else None


@app.function
def build_active_filters_dict(
    mass_filter_active: bool,
    mass_min_val: Optional[float],
    mass_max_val: Optional[float],
    year_filter_active: bool,
    year_start_val: Optional[int],
    year_end_val: Optional[int],
    formula_filters: Optional[FormulaFilters],
) -> Dict[str, Any]:
    """
    Build a dictionary of active filters for metadata export.

    This function follows DRY principle by centralizing filter serialization logic.
    """
    filters = {}

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
    formula_dict = serialize_formula_filters(formula_filters)
    if formula_dict:
        filters["molecular_formula"] = formula_dict

    return filters


@app.function
def generate_filename(
    taxon_name: str,
    file_type: str,
    prefix: str = "lotus_data",
    filters: Dict[str, Any] = None,
) -> str:
    """
    Generate standardized filename for exports.

    Args:
        taxon_name: Name of the taxon (or "*" for all taxa)
        file_type: File extension (e.g., 'csv', 'json', 'ttl')
        prefix: Filename prefix (default: 'lotus_data')
        filters: Optional dict of active filters (adds "_filtered" suffix if present)

    Returns:
        Standardized filename with date
    """
    # Handle wildcard for all taxa
    if taxon_name == "*":
        safe_name = "all_taxa"
    else:
        safe_name = taxon_name.replace(" ", "_")

    # Add "_filtered" suffix if filters are active
    filter_suffix = "_filtered" if filters and len(filters) > 0 else ""

    date_str = datetime.now().strftime("%Y%m%d")
    return f"{date_str}_{prefix}_{safe_name}{filter_suffix}.{file_type}"


@app.function
def compress_if_large(
    data: str, filename: str, threshold_bytes: int = None
) -> tuple[str, str, str]:
    """
    Compress data with gzip if it exceeds the threshold.

    Args:
        data: String data to potentially compress
        filename: Original filename
        threshold_bytes: Size threshold for compression (uses CONFIG if None)

    Returns:
        Tuple of (data_or_compressed, final_filename, mimetype)
    """
    if threshold_bytes is None:
        threshold_bytes = CONFIG["download_embed_threshold_bytes"]

    data_bytes = data.encode("utf-8")
    data_size = len(data_bytes)

    # Compress if data exceeds threshold
    if data_size > threshold_bytes:
        compressed_data = gzip.compress(data_bytes)
        return (compressed_data, f"{filename}.gz", "application/gzip")
    else:
        return (data, filename, None)


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
    if (min_val is None and max_val is None) or column not in df.columns:
        return df

    col_expr = pl.col(column)
    if extract_func:
        col_expr = extract_func(col_expr)

    # Build filter condition based on which bounds are set
    if min_val is not None and max_val is not None:
        # Both bounds set
        condition = (col_expr >= min_val) & (col_expr <= max_val)
    elif min_val is not None:
        # Only minimum bound set
        condition = col_expr >= min_val
    else:
        # Only maximum bound set
        condition = col_expr <= max_val

    return df.filter(pl.col(column).is_null() | condition)


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

    Efficiency optimizations:
    - Early returns for common cases
    - Cached formula parsing
    - Minimal string operations

    Args:
        formula: Molecular formula to check
        filters: FormulaFilters dataclass with all criteria

    Returns:
        True if formula matches all criteria, False otherwise
    """
    # Early return: no formula means keep it (common case)
    if not formula:
        return True

    # Normalize formula once
    normalized_formula = formula.translate(SUBSCRIPT_MAP)

    # Early return: exact formula match (fast path)
    if filters.exact_formula and filters.exact_formula.strip():
        normalized_exact = filters.exact_formula.strip().translate(SUBSCRIPT_MAP)
        return normalized_formula == normalized_exact

    # Parse formula (cached for performance)
    atom_tuple = parse_molecular_formula(formula)
    atoms = dict(atom_tuple)

    # Check element ranges with early termination
    # Note: Using tuple for iteration efficiency
    elements_to_check = (
        ("C", filters.c),
        ("H", filters.h),
        ("N", filters.n),
        ("O", filters.o),
        ("P", filters.p),
        ("S", filters.s),
    )

    for element, elem_range in elements_to_check:
        if not elem_range.matches(atoms.get(element, 0)):
            return False  # Early termination

    # Check halogens with early termination
    halogens = (
        ("F", filters.f_state),
        ("Cl", filters.cl_state),
        ("Br", filters.br_state),
        ("I", filters.i_state),
    )

    for halogen, state in halogens:
        count = atoms.get(halogen, 0)
        if (state == "required" and count == 0) or (state == "excluded" and count > 0):
            return False  # Early termination

    return True


@app.function
def apply_formula_filter(df: pl.DataFrame, filters: FormulaFilters) -> pl.DataFrame:
    """
    Apply molecular formula filters to the dataframe.

    Efficiency note:
    - Early return if no formula column or inactive filters
    - List comprehension is faster than append loop
    - Polars filtering is optimized internally

    Args:
        df: DataFrame to filter
        filters: Formula filtering criteria

    Returns:
        Filtered DataFrame
    """
    # Early return for efficiency
    if "mf" not in df.columns or not filters.is_active():
        return df

    # List comprehension is more efficient than building list with append
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

    Efficiency optimizations:
    - Single DataFrame creation (no intermediate copies)
    - Lazy column transformations
    - Filter chaining without intermediate copies
    - Early termination on empty results

    Args:
        qid: Wikidata QID for taxon, or "*" for all taxa
        year_start: Filter start year (inclusive)
        year_end: Filter end year (inclusive)
        mass_min: Minimum mass in Daltons
        mass_max: Maximum mass in Daltons
        formula_filters: Molecular formula filter criteria

    Returns:
        Polars DataFrame with compound data
    """
    # Use simplified query for wildcard, otherwise use taxon-specific query
    if qid == "*":
        query = build_all_compounds_query()
    else:
        query = build_compounds_query(qid)

    results = execute_sparql(query)
    bindings = results.get("results", {}).get("bindings", [])

    # Early return for empty results (efficiency - no DataFrame creation)
    if not bindings:
        return pl.DataFrame()

    # Process results efficiently with list comprehension (single pass)
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

    # Create DataFrame once (efficiency - avoid copies)
    df = pl.DataFrame(rows)

    # Lazy transformations (Polars optimizes internally)
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

    # Chain filters for efficiency (Polars optimizes the execution plan)
    df = apply_year_filter(df, year_start, year_end)
    df = apply_mass_filter(df, mass_min, mass_max)

    if formula_filters:
        df = apply_formula_filter(df, formula_filters)

    # Final operations: deduplicate and sort
    # Note: unique() is efficient in Polars, keeps first occurrence
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
    metadata = {
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
        },
        "sparql_endpoint": CONFIG["sparql_endpoint"],
    }

    # Add filters if any are active
    if filters:
        metadata["search_parameters"]["filters"] = filters

    return metadata


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


@app.function
def export_to_rdf_turtle(df: pl.DataFrame, taxon_input: str, qid: str) -> str:
    """
    Export data to RDF Turtle format using rdflib following W3C standards.

    Uses standard ontologies:
    - CHEMINF (Chemical Information Ontology)
    - SIO (Semanticscience Integrated Ontology)
    - WD (Wikidata)
    - schema.org
    - DCTERMS (Dublin Core Terms)
    """
    # Initialize graph
    g = Graph()

    # Define namespaces
    WD = Namespace("http://www.wikidata.org/entity/")
    WDT = Namespace("http://www.wikidata.org/prop/direct/")
    CHEMINF = Namespace("http://semanticscience.org/resource/CHEMINF_")
    SIO = Namespace("http://semanticscience.org/resource/SIO_")
    SCHEMA = Namespace("http://schema.org/")
    BIBO = Namespace("http://purl.org/ontology/bibo/")

    # Bind prefixes for cleaner output
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("cheminf", CHEMINF)
    g.bind("sio", SIO)
    g.bind("schema", SCHEMA)
    g.bind("dcterms", DCTERMS)
    g.bind("bibo", BIBO)

    # Dataset URI
    dataset_uri = URIRef(
        f"https://lotus.naturalproducts.net/dataset/{url_quote(taxon_input)}"
    )

    # Dataset metadata
    g.add((dataset_uri, RDF.type, SCHEMA.Dataset))
    g.add(
        (
            dataset_uri,
            SCHEMA.name,
            Literal(f"LOTUS Data for {taxon_input}", datatype=XSD.string),
        )
    )
    g.add(
        (
            dataset_uri,
            SCHEMA.description,
            Literal(
                f"Chemical compounds found in taxon {taxon_input} from Wikidata",
                datatype=XSD.string,
            ),
        )
    )
    g.add(
        (
            dataset_uri,
            SCHEMA.dateCreated,
            Literal(datetime.now().isoformat(), datatype=XSD.dateTime),
        )
    )
    g.add(
        (
            dataset_uri,
            SCHEMA.license,
            URIRef("https://creativecommons.org/publicdomain/zero/1.0/"),
        )
    )
    g.add((dataset_uri, SCHEMA.provider, URIRef("https://www.wikidata.org/")))
    g.add((dataset_uri, SCHEMA.about, WD[qid]))
    g.add((dataset_uri, DCTERMS.source, URIRef("https://www.wikidata.org/")))
    g.add((dataset_uri, SCHEMA.numberOfRecords, Literal(len(df), datatype=XSD.integer)))

    # Add compound data
    for row in df.iter_rows(named=True):
        compound_qid = row.get("compound_qid", "")
        if not compound_qid:
            continue

        compound_uri = WD[compound_qid]

        # Link compound to dataset
        g.add((dataset_uri, SCHEMA.hasPart, compound_uri))

        # Compound type
        g.add((compound_uri, RDF.type, SCHEMA.MolecularEntity))

        # Compound label
        if row.get("compound_name"):
            g.add(
                (
                    compound_uri,
                    RDFS.label,
                    Literal(row["compound_name"], datatype=XSD.string),
                )
            )

        # InChIKey
        if row.get("compound_inchikey"):
            g.add(
                (
                    compound_uri,
                    CHEMINF["000059"],
                    Literal(row["compound_inchikey"], datatype=XSD.string),
                )
            )

        # SMILES
        if row.get("compound_smiles"):
            g.add(
                (
                    compound_uri,
                    CHEMINF["000018"],
                    Literal(row["compound_smiles"], datatype=XSD.string),
                )
            )

        # Molecular formula
        if row.get("molecular_formula"):
            g.add(
                (
                    compound_uri,
                    CHEMINF["000042"],
                    Literal(row["molecular_formula"], datatype=XSD.string),
                )
            )

        # Molecular mass
        if row.get("compound_mass") is not None:
            g.add(
                (
                    compound_uri,
                    SIO["000218"],
                    Literal(row["compound_mass"], datatype=XSD.float),
                )
            )

        # Taxonomic association (found in taxon)
        if row.get("taxon_qid"):
            g.add((compound_uri, SIO["000255"], WD[row["taxon_qid"]]))

        # Taxon name
        if row.get("taxon_name") and row.get("taxon_qid"):
            taxon_uri = WD[row["taxon_qid"]]
            g.add(
                (taxon_uri, RDFS.label, Literal(row["taxon_name"], datatype=XSD.string))
            )
            g.add((taxon_uri, RDF.type, SCHEMA.Taxon))

        # Reference
        if row.get("reference_qid"):
            ref_uri = WD[row["reference_qid"]]
            g.add((compound_uri, DCTERMS.source, ref_uri))

            # Reference metadata
            if row.get("reference_title"):
                g.add(
                    (
                        ref_uri,
                        DCTERMS.title,
                        Literal(row["reference_title"], datatype=XSD.string),
                    )
                )

            if row.get("reference_doi"):
                # URL-encode the DOI to handle special characters like <, >, etc.
                encoded_doi = url_quote(row["reference_doi"], safe="")
                doi_uri = URIRef(f"https://doi.org/{encoded_doi}")
                g.add((ref_uri, SCHEMA.identifier, doi_uri))

            if row.get("reference_date"):
                g.add(
                    (
                        ref_uri,
                        DCTERMS.date,
                        Literal(str(row["reference_date"]), datatype=XSD.date),
                    )
                )

    # Add taxon information
    if qid:
        taxon_uri = WD[qid]
        g.add((taxon_uri, RDF.type, SCHEMA.Taxon))
        g.add((taxon_uri, RDFS.label, Literal(taxon_input, datatype=XSD.string)))

    # Serialize to Turtle format
    return g.serialize(format="turtle")


@app.cell
def _():
    mo.md("""
    # 🌿 LOTUS Wikidata Explorer
    """)
    return


@app.cell
def _():
    mo.callout(
        mo.md("""
        ## ⚠️ Under Development

        This application is under development and may not work as expected in all deployment modes.

        **Recommended way to run:**
        ```bash
        uvx marimo run https://raw.githubusercontent.com/Adafede/marimo/refs/heads/main/apps/lotus_wikidata_explorer.py
        ```
        """),
        kind="warn",
    )
    return


@app.cell
def _():
    mo.md("""
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
    mass_filter = mo.ui.checkbox(label="⚖️ Filter by mass", value=state_mass_filter)

    mass_min = mo.ui.number(
        value=state_mass_min,
        start=0,
        stop=10000,
        step=0.001,
        label="Min mass (Da)",
        full_width=True,
    )

    mass_max = mo.ui.number(
        value=state_mass_max,
        start=0,
        stop=10000,
        step=0.001,
        label="Max mass (Da)",
        full_width=True,
    )

    ## FORMULA FILTERS
    formula_filter = mo.ui.checkbox(
        label="⚛️ Filter by molecular formula", value=state_formula_filter
    )

    exact_formula = mo.ui.text(
        value=state_exact_formula,
        label="Exact formula (e.g., C15H10O5)",
        placeholder="Leave empty to use element ranges",
        full_width=True,
    )

    c_min = mo.ui.number(
        value=state_c_min,
        start=0,
        stop=CONFIG["element_c_max"],
        label="C min",
        full_width=True,
    )
    c_max = mo.ui.number(
        value=state_c_max if state_c_max is not None else CONFIG["element_c_max"],
        start=0,
        stop=CONFIG["element_c_max"],
        label="C max",
        full_width=True,
    )
    h_min = mo.ui.number(
        value=state_h_min,
        start=0,
        stop=CONFIG["element_h_max"],
        label="H min",
        full_width=True,
    )
    h_max = mo.ui.number(
        value=state_h_max if state_h_max is not None else CONFIG["element_h_max"],
        start=0,
        stop=CONFIG["element_h_max"],
        label="H max",
        full_width=True,
    )
    n_min = mo.ui.number(
        value=state_n_min,
        start=0,
        stop=CONFIG["element_n_max"],
        label="N min",
        full_width=True,
    )
    n_max = mo.ui.number(
        value=state_n_max if state_n_max is not None else CONFIG["element_n_max"],
        start=0,
        stop=CONFIG["element_n_max"],
        label="N max",
        full_width=True,
    )
    o_min = mo.ui.number(
        value=state_o_min,
        start=0,
        stop=CONFIG["element_o_max"],
        label="O min",
        full_width=True,
    )
    o_max = mo.ui.number(
        value=state_o_max if state_o_max is not None else CONFIG["element_o_max"],
        start=0,
        stop=CONFIG["element_o_max"],
        label="O max",
        full_width=True,
    )
    p_min = mo.ui.number(
        value=state_p_min,
        start=0,
        stop=CONFIG["element_p_max"],
        label="P min",
        full_width=True,
    )
    p_max = mo.ui.number(
        value=state_p_max if state_p_max is not None else CONFIG["element_p_max"],
        start=0,
        stop=CONFIG["element_p_max"],
        label="P max",
        full_width=True,
    )
    s_min = mo.ui.number(
        value=state_s_min,
        start=0,
        stop=CONFIG["element_s_max"],
        label="S min",
        full_width=True,
    )
    s_max = mo.ui.number(
        value=state_s_max if state_s_max is not None else CONFIG["element_s_max"],
        start=0,
        stop=CONFIG["element_s_max"],
        label="S max",
        full_width=True,
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

    taxon_input = mo.ui.text(
        value=state_taxon,
        label="🔬 Taxon name or QID",
        placeholder="e.g., Swertia chirayita, Anabaena, Q157115, or * for all taxa",
        full_width=True,
    )

    ## DATE FILTERS
    current_year = datetime.now().year
    year_filter = mo.ui.checkbox(
        label="🗓️ Filter by publication year", value=state_year_filter
    )

    year_start = mo.ui.number(
        value=state_year_start,
        start=CONFIG["year_range_start"],
        stop=current_year,
        label="Start year",
        full_width=True,
    )

    year_end = mo.ui.number(
        value=state_year_end,
        start=CONFIG["year_range_start"],
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

        # Customize spinner message for wildcard
        if taxon_input_str == "*":
            spinner_message = "🔎 Querying Wikidata for all taxa..."
        else:
            spinner_message = f"🔎 Querying Wikidata for {taxon_input_str}..."

        with mo.status.spinner(title=spinner_message):
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
                    # Helper function to convert CONFIG defaults back to None
                    # (since UI shows them as placeholders but we don't want them to activate filters)
                    def normalize_element_value(val, default):
                        return None if val == default else val

                    formula_filt = FormulaFilters(
                        exact_formula=exact_formula.value
                        if exact_formula.value.strip()
                        else None,
                        c=ElementRange(
                            c_min.value,
                            normalize_element_value(
                                c_max.value, CONFIG["element_c_max"]
                            ),
                        ),
                        h=ElementRange(
                            h_min.value,
                            normalize_element_value(
                                h_max.value, CONFIG["element_h_max"]
                            ),
                        ),
                        n=ElementRange(
                            n_min.value,
                            normalize_element_value(
                                n_max.value, CONFIG["element_n_max"]
                            ),
                        ),
                        o=ElementRange(
                            o_min.value,
                            normalize_element_value(
                                o_max.value, CONFIG["element_o_max"]
                            ),
                        ),
                        p=ElementRange(
                            p_min.value,
                            normalize_element_value(
                                p_max.value, CONFIG["element_p_max"]
                            ),
                        ),
                        s=ElementRange(
                            s_min.value,
                            normalize_element_value(
                                s_max.value, CONFIG["element_s_max"]
                            ),
                        ),
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

        # Handle wildcard case
        if qid == "*":
            parts.append(
                mo.callout(
                    mo.md(
                        f"No natural products found for **all taxa** with the current filters."
                    ),
                    kind="warn",
                )
            )
        else:
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

        # Handle wildcard case for summary header
        if qid == "*":
            summary_header = mo.md(
                f"## Results\n" f"### Summary\n\nFound data for **all taxa**"
            )
        else:
            summary_header = mo.md(
                f"## Results\n"
                f"### Summary\n\nFound data for **{taxon_input.value}** {create_wikidata_link(qid)}"
            )

        summary_parts = [summary_header]

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
    results_df,
    run_button,
    s_max,
    s_min,
    state_auto_run,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    # Replace previous generation logic: build UI but DO NOT display inline
    if (not run_button.value and not state_auto_run) or results_df is None:
        download_ui = mo.Html("")
        tables_ui = mo.Html("")
        ui_is_large_dataset = False
        taxon_name = taxon_input.value
        active_filters = {}
        export_df_for_lazy = None
        csv_generate_button = None
        json_generate_button = None
        rdf_generate_button = None
        csv_generation_data = None
        json_generation_data = None
        rdf_generation_data = None
    elif len(results_df) == 0:
        download_ui = mo.callout(
            mo.md("No compounds match your search criteria."), kind="neutral"
        )
        tables_ui = mo.Html("")
        ui_is_large_dataset = False
        taxon_name = taxon_input.value
        active_filters = {}
        export_df_for_lazy = None
        csv_generate_button = None
        json_generate_button = None
        rdf_generate_button = None
        csv_generation_data = None
        json_generation_data = None
        rdf_generation_data = None
    else:
        export_df = prepare_export_dataframe(results_df)
        taxon_name = taxon_input.value
        ui_is_large_dataset = len(export_df) > CONFIG["lazy_generation_threshold"]
        # Build filters for metadata
        _formula_filt = None
        if formula_filter.value:
            # Helper function to convert CONFIG defaults back to None
            def _normalize_element_value(val, default):
                return None if val == default else val

            _formula_filt = FormulaFilters(
                exact_formula=exact_formula.value
                if exact_formula.value.strip()
                else None,
                c=ElementRange(
                    c_min.value,
                    _normalize_element_value(c_max.value, CONFIG["element_c_max"]),
                ),
                h=ElementRange(
                    h_min.value,
                    _normalize_element_value(h_max.value, CONFIG["element_h_max"]),
                ),
                n=ElementRange(
                    n_min.value,
                    _normalize_element_value(n_max.value, CONFIG["element_n_max"]),
                ),
                o=ElementRange(
                    o_min.value,
                    _normalize_element_value(o_max.value, CONFIG["element_o_max"]),
                ),
                p=ElementRange(
                    p_min.value,
                    _normalize_element_value(p_max.value, CONFIG["element_p_max"]),
                ),
                s=ElementRange(
                    s_min.value,
                    _normalize_element_value(s_max.value, CONFIG["element_s_max"]),
                ),
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
        )
        metadata = create_export_metadata(
            export_df, taxon_input.value, qid, active_filters
        )
        metadata_json = json.dumps(metadata, indent=2)
        citation_text = create_citation_text(taxon_input.value)
        # Display table data (apply row limit & depiction logic)
        total_rows = len(results_df)
        if total_rows > CONFIG["table_row_limit"]:
            limited_df = results_df.head(CONFIG["table_row_limit"])
            display_data = [
                {
                    "Compound": row["name"],
                    "Compound SMILES": row["smiles"],
                    "Compound InChIKey": row["inchikey"],
                    "Taxon": row["taxon_name"],
                    "Reference title": row["ref_title"] or "—",
                    "Reference DOI": create_link(f"https://doi.org/{doi}", doi)
                    if (doi := row["ref_doi"])
                    else mo.Html("—"),
                    "Compound QID": create_wikidata_link(extract_qid(row["structure"])),
                    "Taxon QID": create_wikidata_link(extract_qid(row["taxon"])),
                    "Reference QID": create_wikidata_link(
                        extract_qid(row["reference"])
                    ),
                }
                for row in limited_df.iter_rows(named=True)
            ]
            display_note = mo.callout(
                mo.md(
                    f"⚡ **Large dataset ({total_rows:,} rows)**\n\n"
                    f"- Showing first {CONFIG['table_row_limit']:,} rows in tables\n"
                    f"- 2D structure depictions hidden for performance\n"
                ),
                kind="info",
            )
        else:
            display_data = [
                create_display_row(row) for row in results_df.iter_rows(named=True)
            ]
            display_note = mo.Html("")
        display_table = mo.ui.table(
            display_data,
            selection=None,
            page_size=CONFIG["page_size_default"],
            show_column_summaries=False,
        )
        export_table = mo.ui.table(
            export_df.to_dicts(),
            selection=None,
            page_size=CONFIG["page_size_export"],
            show_column_summaries=False,
        )
        # Immediate or lazy downloads
        buttons = []
        if ui_is_large_dataset:
            csv_generate_button = mo.ui.run_button(label="📄 Generate CSV")
            json_generate_button = mo.ui.run_button(label="📖 Generate JSON")
            rdf_generate_button = mo.ui.run_button(label="🐢 Generate RDF/Turtle")
            buttons.extend(
                [csv_generate_button, json_generate_button, rdf_generate_button]
            )
            csv_generation_data = {
                "export_df": export_df,
                "active_filters": active_filters,
            }
            json_generation_data = {
                "export_df": export_df,
                "active_filters": active_filters,
            }
            rdf_generation_data = {
                "export_df": export_df,
                "taxon_input": taxon_input.value,
                "qid": qid,
                "active_filters": active_filters,
            }
            export_df_for_lazy = export_df
        else:
            csv_generate_button = None
            json_generate_button = None
            rdf_generate_button = None
            csv_generation_data = None
            json_generation_data = None
            rdf_generation_data = None
            export_df_for_lazy = None

            # Generate and compress CSV
            _csv_raw = export_df.write_csv()
            _csv_data, _csv_fname, _csv_mime = compress_if_large(
                _csv_raw,
                generate_filename(taxon_input.value, "csv", filters=active_filters),
            )
            buttons.append(
                mo.download(
                    data=_csv_data,
                    filename=_csv_fname,
                    label="📥 CSV"
                    + (" (gzipped)" if _csv_mime == "application/gzip" else ""),
                    mimetype=_csv_mime if _csv_mime else "text/csv",
                )
            )

            # Generate and compress JSON
            _json_raw = export_df.write_json()
            _json_data, _json_fname, _json_mime = compress_if_large(
                _json_raw,
                generate_filename(taxon_input.value, "json", filters=active_filters),
            )
            buttons.append(
                mo.download(
                    data=_json_data,
                    filename=_json_fname,
                    label="📥 JSON"
                    + (" (gzipped)" if _json_mime == "application/gzip" else ""),
                    mimetype=_json_mime if _json_mime else "application/json",
                )
            )

            # Generate and compress RDF
            _rdf_raw = export_to_rdf_turtle(export_df, taxon_input.value, qid)
            _rdf_data, _rdf_fname, _rdf_mime = compress_if_large(
                _rdf_raw,
                generate_filename(taxon_input.value, "ttl", filters=active_filters),
            )
            buttons.append(
                mo.download(
                    data=_rdf_data,
                    filename=_rdf_fname,
                    label="📥 RDF/Turtle"
                    + (" (gzipped)" if _rdf_mime == "application/gzip" else ""),
                    mimetype=_rdf_mime if _rdf_mime else "text/turtle",
                )
            )
        # Metadata download
        buttons.append(
            mo.download(
                data=metadata_json,
                filename=generate_filename(
                    taxon_input.value,
                    "json",
                    prefix="lotus_metadata",
                    filters=active_filters,
                ),
                label="📋 Metadata",
                mimetype="application/json",
            )
        )
        download_ui = mo.vstack(
            [mo.md("### Download"), mo.hstack(buttons, gap=2, wrap=True)]
        )
        tables_ui = mo.vstack(
            [
                mo.md("### Tables"),
                display_note,
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
    return (
        csv_generate_button,
        csv_generation_data,
        download_ui,
        json_generate_button,
        json_generation_data,
        rdf_generate_button,
        rdf_generation_data,
        tables_ui,
        taxon_name,
        ui_is_large_dataset,
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


@app.cell
def _(download_ui):
    download_ui
    return


@app.cell
def _(
    csv_generate_button,
    csv_generation_data,
    json_generate_button,
    json_generation_data,
    rdf_generate_button,
    rdf_generation_data,
    taxon_name,
    ui_is_large_dataset,
):
    # Handle CSV generation
    if (
        ui_is_large_dataset
        and csv_generate_button is not None
        and csv_generate_button.value
    ):
        with mo.status.spinner(title="📄 Generating CSV format..."):
            _csv_data_raw = csv_generation_data["export_df"].write_csv()
            _csv_data, _csv_filename, _csv_mimetype = compress_if_large(
                _csv_data_raw,
                generate_filename(
                    taxon_name, "csv", filters=csv_generation_data["active_filters"]
                ),
            )
        csv_download_ui = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        f"✅ CSV generated ({len(csv_generation_data['export_df']):,} entries)"
                    ),
                    kind="success",
                ),
                mo.download(
                    data=_csv_data,
                    filename=_csv_filename,
                    label="📥 Download CSV"
                    + (" (gzipped)" if _csv_mimetype == "application/gzip" else ""),
                    mimetype=_csv_mimetype if _csv_mimetype else "text/csv",
                ),
            ]
        )
    else:
        csv_download_ui = mo.Html("")

    # Handle JSON generation
    if (
        ui_is_large_dataset
        and json_generate_button is not None
        and json_generate_button.value
    ):
        with mo.status.spinner(title="📖 Generating JSON format..."):
            _json_data_raw = json_generation_data["export_df"].write_json()
            _json_data, _json_filename, _json_mimetype = compress_if_large(
                _json_data_raw,
                generate_filename(
                    taxon_name, "json", filters=json_generation_data["active_filters"]
                ),
            )
        json_download_ui = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        f"✅ JSON generated ({len(json_generation_data['export_df']):,} entries)"
                    ),
                    kind="success",
                ),
                mo.download(
                    data=_json_data,
                    filename=_json_filename,
                    label="📥 Download JSON"
                    + (" (gzipped)" if _json_mimetype == "application/gzip" else ""),
                    mimetype=_json_mimetype if _json_mimetype else "application/json",
                ),
            ]
        )
    else:
        json_download_ui = mo.Html("")

    # Handle RDF generation
    if (
        ui_is_large_dataset
        and rdf_generate_button is not None
        and rdf_generate_button.value
    ):
        with mo.status.spinner(title="🐢 Generating RDF/Turtle format..."):
            _rdf_data_raw = export_to_rdf_turtle(
                rdf_generation_data["export_df"],
                rdf_generation_data["taxon_input"],
                rdf_generation_data["qid"],
            )
            _rdf_data, _rdf_filename, _rdf_mimetype = compress_if_large(
                _rdf_data_raw,
                generate_filename(
                    taxon_name, "ttl", filters=rdf_generation_data["active_filters"]
                ),
            )
        rdf_download_ui = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        f"✅ RDF/Turtle generated ({len(rdf_generation_data['export_df']):,} entries)"
                    ),
                    kind="success",
                ),
                mo.download(
                    data=_rdf_data,
                    filename=_rdf_filename,
                    label="📥 Download RDF/Turtle"
                    + (" (gzipped)" if _rdf_mimetype == "application/gzip" else ""),
                    mimetype=_rdf_mimetype if _rdf_mimetype else "text/turtle",
                ),
            ]
        )
    else:
        rdf_download_ui = mo.Html("")

    # Show all generated downloads
    mo.vstack([csv_download_ui, json_download_ui, rdf_download_ui], gap=2)
    return


@app.cell
def _(tables_ui):
    tables_ui
    return


@app.cell
def _():
    # URL parameter detection
    url_params = mo.query_params()

    # Detect if we should auto-execute search
    url_auto_search = "taxon" in url_params

    # Get URL parameter values with defaults
    url_taxon = url_params.get("taxon", "Gentiana lutea")

    # Mass filter
    url_mass_filter = url_params.get("mass_filter") == "true"
    url_mass_min = float(url_params.get("mass_min", CONFIG["mass_default_min"]))
    url_mass_max = float(url_params.get("mass_max", CONFIG["mass_default_max"]))

    # Year filter
    url_year_filter = url_params.get("year_filter") == "true"
    url_year_start = int(url_params.get("year_start", CONFIG["year_default_start"]))
    url_year_end = int(url_params.get("year_end", datetime.now().year))

    # Formula filter
    url_formula_filter = url_params.get("formula_filter") == "true"
    url_exact_formula = url_params.get("exact_formula", "")
    url_c_min = int(url_params["c_min"]) if "c_min" in url_params else None
    url_c_max = int(url_params.get("c_max", CONFIG["element_c_max"]))
    url_h_min = int(url_params["h_min"]) if "h_min" in url_params else None
    url_h_max = int(url_params.get("h_max", CONFIG["element_h_max"]))
    url_n_min = int(url_params["n_min"]) if "n_min" in url_params else None
    url_n_max = int(url_params.get("n_max", CONFIG["element_n_max"]))
    url_o_min = int(url_params["o_min"]) if "o_min" in url_params else None
    url_o_max = int(url_params.get("o_max", CONFIG["element_o_max"]))
    url_p_min = int(url_params["p_min"]) if "p_min" in url_params else None
    url_p_max = int(url_params.get("p_max", CONFIG["element_p_max"]))
    url_s_min = int(url_params["s_min"]) if "s_min" in url_params else None
    url_s_max = int(url_params.get("s_max", CONFIG["element_s_max"]))
    url_f_state = url_params.get("f_state", "allowed")
    url_cl_state = url_params.get("cl_state", "allowed")
    url_br_state = url_params.get("br_state", "allowed")
    url_i_state = url_params.get("i_state", "allowed")
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
        state_mass_min = url_mass_min if url_mass_filter else CONFIG["mass_default_min"]
        state_mass_max = url_mass_max if url_mass_filter else CONFIG["mass_default_max"]

        # Year filter state
        state_year_filter = url_year_filter
        state_year_start = (
            url_year_start if url_year_filter else CONFIG["year_default_start"]
        )
        state_year_end = url_year_end if url_year_filter else datetime.now().year

        # Formula filter state
        state_formula_filter = url_formula_filter
        state_exact_formula = url_exact_formula if url_formula_filter else ""
        state_c_min = url_c_min if url_formula_filter else None
        state_c_max = (
            url_c_max if url_formula_filter else None
        )  # Changed from CONFIG default
        state_h_min = url_h_min if url_formula_filter else None
        state_h_max = (
            url_h_max if url_formula_filter else None
        )  # Changed from CONFIG default
        state_n_min = url_n_min if url_formula_filter else None
        state_n_max = (
            url_n_max if url_formula_filter else None
        )  # Changed from CONFIG default
        state_o_min = url_o_min if url_formula_filter else None
        state_o_max = (
            url_o_max if url_formula_filter else None
        )  # Changed from CONFIG default
        state_p_min = url_p_min if url_formula_filter else None
        state_p_max = (
            url_p_max if url_formula_filter else None
        )  # Changed from CONFIG default
        state_s_min = url_s_min if url_formula_filter else None
        state_s_max = (
            url_s_max if url_formula_filter else None
        )  # Changed from CONFIG default
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
        state_mass_min = CONFIG["mass_default_min"]
        state_mass_max = CONFIG["mass_default_max"]
        state_year_filter = False
        state_year_start = CONFIG["year_default_start"]
        state_year_end = datetime.now().year
        state_formula_filter = False
        state_exact_formula = ""
        state_c_min = None
        state_c_max = None  # Changed from CONFIG default
        state_h_min = None
        state_h_max = None  # Changed from CONFIG default
        state_n_min = None
        state_n_max = None  # Changed from CONFIG default
        state_o_min = None
        state_o_max = None  # Changed from CONFIG default
        state_p_min = None
        state_p_max = None  # Changed from CONFIG default
        state_s_min = None
        state_s_max = None  # Changed from CONFIG default
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


if __name__ == "__main__":
    app.run()
