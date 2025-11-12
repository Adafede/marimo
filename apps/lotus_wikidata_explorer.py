# /// script
# requires-python = ">=3.13"
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

    # Set output max bytes safely (deployment environments may have limits)
    try:
        mo._runtime.context.get_context().marimo_config["runtime"][
            "output_max_bytes"
        ] = 1_000_000_000  # 1GB for large datasets
    except Exception:
        # Silently fail if runtime config cannot be set
        pass

    CONFIG = {
        # Application Metadata
        "app_version": "0.0.1",
        "app_name": "LOTUS Wikidata Explorer",
        "app_url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
        # External Services
        "cdk_base": "https://www.simolecule.com/cdkdepict/depict/cot/svg",
        "sparql_endpoint": "https://qlever.cs.uni-freiburg.de/api/wikidata",
        # "sparql_endpoint": "https://query-legacy-full.wikidata.org/sparql",
        "idsm_endpoint": "https://idsm.elixir-czech.cz/sparql/endpoint/",
        "user_agent": "LOTUS Explorer/1.0.0 (https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py)",
        # Network & Performance
        "max_retries": 3,
        "retry_backoff": 2,
        "table_row_limit": 10000,  # Max rows to display (prevents browser slowdown)
        "lazy_generation_threshold": 5000,  # Defer download generation for large datasets
        "download_embed_threshold_bytes": 8_000_000,  # Compress downloads > 8MB
        # UI Styling
        "color_hyperlink": "#006699",
        "page_size_default": 10,  # Rows per page in display table
        "page_size_export": 25,  # Rows per page in export table
        # Filter Defaults
        "year_range_start": 1700,
        "year_default_start": 1900,
        "mass_default_min": 0,
        "mass_default_max": 2000,
        "mass_ui_max": 10000,
        # Element Count Limits (for formula filter UI)
        "element_c_max": 100,
        "element_h_max": 200,
        "element_n_max": 50,
        "element_o_max": 50,
        "element_p_max": 20,
        "element_s_max": 20,
        # Chemical Search
        "default_similarity_threshold": 0.8,  # Tanimoto coefficient threshold
    }

    # Wikidata URLs (constants)
    WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
    WIKIDATA_WIKI_PREFIX = "https://www.wikidata.org/wiki/"

    # ====================================================================
    # ELEMENT CONFIGURATION
    # ====================================================================

    # Element definitions for formula filters (avoid hardcoding element lists)
    ELEMENT_CONFIGS = [
        ("C", "carbon", "element_c_max"),
        ("H", "hydrogen", "element_h_max"),
        ("N", "nitrogen", "element_n_max"),
        ("O", "oxygen", "element_o_max"),
        ("P", "phosphorus", "element_p_max"),
        ("S", "sulfur", "element_s_max"),
    ]

    HALOGEN_CONFIGS = [
        ("F", "fluorine"),
        ("Cl", "chlorine"),
        ("Br", "bromine"),
        ("I", "iodine"),
    ]

    # ====================================================================
    # SPARQL QUERY FRAGMENTS
    # ====================================================================

    # Common SPARQL prefixes
    SPARQL_PREFIXES = """
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    """

    SACHEM_PREFIXES = """
    PREFIX sachem: <http://bioinfo.uochb.cas.cz/rdf/v1.0/sachem#>
    PREFIX idsm: <https://idsm.elixir-czech.cz/sparql/endpoint/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    """

    # Common SELECT clause for compound queries
    COMPOUND_SELECT_VARS = """?compound ?compound_inchikey ?compound_smiles_conn ?taxon_name ?taxon ?ref_qid 
           ?compound_smiles_iso ?compound_mass ?compound_formula ?compoundLabel 
           ?ref_title ?ref_doi ?ref_date"""

    # Common compound identifier retrieval (used in subqueries)
    COMPOUND_IDENTIFIERS = """
      ?compound wdt:P235 ?compound_inchikey;
                wdt:P233 ?compound_smiles_conn.
    """

    # Common taxon-reference association pattern (used in subqueries)
    TAXON_REFERENCE_ASSOCIATION = """
      ?compound p:P703 ?statement.
      ?statement ps:P703 ?taxon;
                 prov:wasDerivedFrom ?ref.
      ?ref pr:P248 ?ref_qid.
      ?taxon wdt:P225 ?taxon_name.
    """

    # Common compound property optionals
    COMPOUND_PROPERTIES_OPTIONAL = """
      OPTIONAL { ?compound wdt:P2017 ?compound_smiles_iso. }
      OPTIONAL { ?compound wdt:P2067 ?compound_mass. }
      OPTIONAL { ?compound wdt:P274 ?compound_formula. }
      OPTIONAL {
        ?compound rdfs:label ?compoundLabel.
        FILTER(LANG(?compoundLabel) = "en")
      }
      OPTIONAL {
        ?compound rdfs:label ?compoundLabel.
        FILTER(LANG(?compoundLabel) = "mul")
      }
    """

    # Common taxonomic and reference optionals
    TAXONOMIC_REFERENCE_OPTIONAL = """
      OPTIONAL {
        ?statement ps:P703 ?taxon;
                   prov:wasDerivedFrom ?ref.
        ?ref pr:P248 ?ref_qid.
        ?compound p:P703 ?statement.
        ?taxon wdt:P225 ?taxon_name.
        OPTIONAL { ?ref_qid wdt:P1476 ?ref_title. }
        OPTIONAL { ?ref_qid wdt:P356 ?ref_doi. }
        OPTIONAL { ?ref_qid wdt:P577 ?ref_date. }
      }
    """

    # Common reference metadata retrieval (after subquery)
    REFERENCE_METADATA_OPTIONAL = """
      OPTIONAL { ?ref_qid wdt:P1476 ?ref_title. }
      OPTIONAL { ?ref_qid wdt:P356 ?ref_doi. }
      OPTIONAL { ?ref_qid wdt:P577 ?ref_date. }
    """

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
    # HTTP SESSION
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
    # DATA CLASSES
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
def build_smiles_substructure_query(smiles: str) -> str:
    """Build SPARQL query for chemical substructure search using SACHEM."""
    return f"""{SPARQL_PREFIXES}{SACHEM_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
      {{
        SELECT ?compound ?compound_inchikey ?compound_smiles_conn WHERE {{
          SERVICE idsm:wikidata {{
            VALUES ?SUBSTRUCTURE {{ "{smiles}" }}
            ?compound sachem:substructureSearch [
              sachem:query ?SUBSTRUCTURE
            ].
          }}
          {COMPOUND_IDENTIFIERS}
        }}
      }}
      {TAXONOMIC_REFERENCE_OPTIONAL}
      {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


@app.function
def build_smiles_similarity_query(smiles: str, threshold: float = 0.8) -> str:
    """Build SPARQL query for chemical similarity search using SACHEM."""
    return f"""{SPARQL_PREFIXES}{SACHEM_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
      {{
        SELECT ?compound ?compound_inchikey ?compound_smiles_conn WHERE {{
          SERVICE idsm:wikidata {{
            VALUES ?QUERY_SMILES {{ "{smiles}" }}
            VALUES ?CUTOFF {{ "{threshold}"^^xsd:double }}
            ?compound sachem:similarCompoundSearch[
            sachem:query ?QUERY_SMILES;
            sachem:cutoff ?CUTOFF
            ].
          }}
          {COMPOUND_IDENTIFIERS}
        }}
      }}
      {TAXONOMIC_REFERENCE_OPTIONAL}
      {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


@app.function
def build_smiles_taxon_query(
    smiles: str, qid: str, search_type: str = "substructure", threshold: float = 0.8
) -> str:
    """Build SPARQL query to find compounds by SMILES within a specific taxon."""
    if search_type == "similarity":
        # Optimized similarity search: SACHEM first, then taxon filter
        return f"""{SPARQL_PREFIXES}{SACHEM_PREFIXES}
        SELECT {COMPOUND_SELECT_VARS} WHERE {{
          {{
            SELECT ?compound ?compound_inchikey ?compound_smiles_conn WHERE {{
              SERVICE idsm:wikidata {{
                VALUES ?QUERY_SMILES {{ "{smiles}" }}
                VALUES ?CUTOFF {{ "{threshold}"^^xsd:double }}
                ?compound sachem:similarCompoundSearch[
                sachem:query ?QUERY_SMILES;
                sachem:cutoff ?CUTOFF
                ].
              }}
              {COMPOUND_IDENTIFIERS}
            }}
          }}
          {TAXON_REFERENCE_ASSOCIATION}
          ?taxon (wdt:P171*) wd:{qid}
          {REFERENCE_METADATA_OPTIONAL}
          {COMPOUND_PROPERTIES_OPTIONAL}
        }}
        """
    else:
        # Optimized substructure search: SACHEM first, then taxon filter
        return f"""{SPARQL_PREFIXES}{SACHEM_PREFIXES}
        SELECT {COMPOUND_SELECT_VARS} WHERE {{
          {{
            SELECT ?compound ?compound_inchikey ?compound_smiles_conn WHERE {{
              SERVICE idsm:wikidata {{
                VALUES ?SUBSTRUCTURE {{ "{smiles}" }}
                ?compound sachem:substructureSearch [
                  sachem:query ?SUBSTRUCTURE
                ].
              }}
              {COMPOUND_IDENTIFIERS}
            }}
          }}
          {TAXON_REFERENCE_ASSOCIATION}
          ?taxon (wdt:P171*) wd:{qid}
          {REFERENCE_METADATA_OPTIONAL}
          {COMPOUND_PROPERTIES_OPTIONAL}
        }}
        """


@app.function
def build_compounds_query(qid: str) -> str:
    """Build SPARQL query to find compounds in a specific taxon and its descendants."""
    return f"""{SPARQL_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
      {{
        SELECT ?compound ?compound_inchikey ?compound_smiles_conn ?taxon ?taxon_name ?ref_qid WHERE {{
          {COMPOUND_IDENTIFIERS}
          {TAXON_REFERENCE_ASSOCIATION}
        }}
      }}
      ?taxon (wdt:P171*) wd:{qid}.
      {REFERENCE_METADATA_OPTIONAL}      
      {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


@app.function
def build_all_compounds_query() -> str:
    """Build SPARQL query to retrieve all compounds."""
    return f"""{SPARQL_PREFIXES}
    SELECT ?compound ?compoundLabel ?compound_inchikey ?compound_smiles_iso 
                    ?compound_smiles_conn ?compound_mass ?compound_formula 
                    ?taxon_name ?taxon ?ref_title ?ref_doi ?ref_qid ?ref_date WHERE {{
      {{
        SELECT ?compound ?compound_inchikey ?compound_smiles_conn ?taxon_name ?taxon ?ref_qid WHERE {{
          {COMPOUND_IDENTIFIERS}
          {TAXON_REFERENCE_ASSOCIATION}
        }}
      }}
      {REFERENCE_METADATA_OPTIONAL}
      {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


@app.function
@lru_cache(maxsize=128)
def execute_sparql(
    query: str, max_retries: int = CONFIG["max_retries"]
) -> Dict[str, Any]:
    """Execute SPARQL query with connection pooling and retry logic."""
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
                timeout=300,
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
                error_detail = ""
                if hasattr(e, "response") and e.response.text:
                    error_detail = f"\nServer response: {e.response.text[:500]}"
                raise Exception(
                    f"🌐 HTTP error {status_code} after {max_retries} attempts.{error_detail}\n"
                    f"💡 Try: Check your internet connection or try again later.\n"
                    f"💡 If this is a deployed environment, the server may have stricter limits."
                )
            time.sleep(CONFIG["retry_backoff"] * (2**attempt))

        except Exception as e:
            if attempt == max_retries - 1:
                query_snippet = query[:200] + "..." if len(query) > 200 else query
                raise Exception(
                    f"❌ Query failed: {str(e)}\n"
                    f"Query snippet: {query_snippet}\n"
                    f"💡 If running in a deployed environment, check timeout and memory limits."
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
    """Build a SPARQL VALUES clause for a list of values."""
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
def build_taxon_connectivity_query(qids: List[str]) -> str:
    """Build query to count compound connections for each taxon."""
    values_clause = build_sparql_values_clause("taxon", qids)
    return f"""
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?taxon (COUNT(DISTINCT ?compound) AS ?compound_count) WHERE {{
      {values_clause}

      # Count compounds directly linked to this taxon or its descendants
      {{
        SELECT ?taxon ?compound WHERE {{
          {values_clause}
          ?descendant (wdt:P171*) ?taxon .
          ?compound wdt:P235 ?inchikey ;
                    p:P703/ps:P703 ?descendant .
        }}
      }}
    }}
    GROUP BY ?taxon
    ORDER BY DESC(?compound_count)
    """


@app.function
def create_taxon_warning_html(
    matches: list, selected_qid: str, is_exact: bool
) -> mo.Html:
    """Create an HTML warning with clickable QID links and taxon details."""
    match_type = "exact matches" if is_exact else "similar taxa"
    intro = (
        f"Ambiguous taxon name. Multiple {match_type} found:"
        if is_exact
        else f"No exact match. Similar taxa found:"
    )

    # Build HTML list of matches
    items = []
    for match_data in matches:
        qid = match_data[0]
        name = match_data[1]
        description = match_data[2] if len(match_data) > 2 else None
        parent = match_data[3] if len(match_data) > 3 else None
        compound_count = match_data[4] if len(match_data) > 4 else None

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
        if compound_count is not None:
            details.append(f"<strong>{compound_count:,} compounds</strong>")

        details_str = " - ".join(details) if details else ""

        # Highlight the selected one
        if qid == selected_qid:
            items.append(
                f"<li>{link} {details_str} <strong>← USING THIS ONE (most compounds)</strong></li>"
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
        <em>For precision, please use a specific QID directly in the search box. When ambiguous, the taxon with the most compound links is automatically selected.</em>
    </div>
    """

    return mo.Html(html)


@app.function
def resolve_taxon_to_qid(taxon_input: str) -> Tuple[Optional[str], Optional[mo.Html]]:
    """Resolve taxon name or QID to a valid QID."""
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

            # Query connectivity to find the most connected taxon
            connectivity_query = build_taxon_connectivity_query(qids)
            connectivity_results = execute_sparql(connectivity_query)
            connectivity_bindings = connectivity_results.get("results", {}).get(
                "bindings", []
            )

            # Build connectivity map
            connectivity_map = {}
            for b in connectivity_bindings:
                qid = extract_qid(get_binding_value(b, "taxon"))
                count = int(get_binding_value(b, "compound_count", "0"))
                connectivity_map[qid] = count

            # Sort exact matches by connectivity (descending)
            sorted_matches = sorted(
                exact_matches, key=lambda x: connectivity_map.get(x[0], 0), reverse=True
            )

            # Get details for display
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

            # Create matches with details including connectivity
            matches_with_details = [
                (
                    qid,
                    name,
                    details_map.get(qid, ("", "", ""))[1],
                    details_map.get(qid, ("", "", ""))[2],
                    connectivity_map.get(qid, 0),
                )
                for qid, name in sorted_matches
            ]

            # Use the most connected taxon (first in sorted list)
            selected_qid = sorted_matches[0][0]
            warning_html = create_taxon_warning_html(
                matches_with_details, selected_qid, is_exact=True
            )
            return selected_qid, warning_html

        # No exact match - use first result with warning
        if len(matches) > 1:
            # Get details for similar matches (limit to 5)
            qids = [qid for qid, _ in matches[:5]]

            # Query connectivity to find the most connected taxon
            connectivity_query = build_taxon_connectivity_query(qids)
            connectivity_results = execute_sparql(connectivity_query)
            connectivity_bindings = connectivity_results.get("results", {}).get(
                "bindings", []
            )

            # Build connectivity map
            connectivity_map = {}
            for b in connectivity_bindings:
                qid = extract_qid(get_binding_value(b, "taxon"))
                count = int(get_binding_value(b, "compound_count", "0"))
                connectivity_map[qid] = count

            # Sort matches by connectivity (descending)
            sorted_matches = sorted(
                matches[:5], key=lambda x: connectivity_map.get(x[0], 0), reverse=True
            )

            # Get details for display
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

            # Create matches with details including connectivity
            matches_with_details = [
                (
                    qid,
                    name,
                    details_map.get(qid, ("", "", ""))[1],
                    details_map.get(qid, ("", "", ""))[2],
                    connectivity_map.get(qid, 0),
                )
                for qid, name in sorted_matches
            ]

            # Use the most connected taxon (first in sorted list)
            selected_qid = sorted_matches[0][0]
            warning_html = create_taxon_warning_html(
                matches_with_details, selected_qid, is_exact=False
            )
            return selected_qid, warning_html

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
    return create_link(f"{WIKIDATA_WIKI_PREFIX}{qid}", qid) if qid else mo.Html("-")


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
    element_attrs = [filters.c, filters.h, filters.n, filters.o, filters.p, filters.s]
    for (_, element_name, _), element_range in zip(ELEMENT_CONFIGS, element_attrs):
        range_dict = serialize_element_range(element_range)
        if range_dict:
            result[element_name] = range_dict

    # Halogen states
    halogen_attrs = [
        filters.f_state,
        filters.cl_state,
        filters.br_state,
        filters.i_state,
    ]
    halogen_states = {}
    for (_, halogen_name), state in zip(HALOGEN_CONFIGS, halogen_attrs):
        if state != "allowed":
            halogen_states[halogen_name] = state

    if halogen_states:
        result["halogens"] = halogen_states

    return result if result else None


@app.function
def normalize_element_value(val: int, default: int) -> Optional[int]:
    """Normalize element value by converting default values to None."""
    return None if val == default else val


@app.function
def create_formula_filters(
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
) -> FormulaFilters:
    """Factory function to create FormulaFilters from UI values."""
    return FormulaFilters(
        exact_formula=exact_formula.strip()
        if exact_formula and exact_formula.strip()
        else None,
        c=ElementRange(
            c_min,
            normalize_element_value(c_max, CONFIG["element_c_max"]),
        ),
        h=ElementRange(
            h_min,
            normalize_element_value(h_max, CONFIG["element_h_max"]),
        ),
        n=ElementRange(
            n_min,
            normalize_element_value(n_max, CONFIG["element_n_max"]),
        ),
        o=ElementRange(
            o_min,
            normalize_element_value(o_max, CONFIG["element_o_max"]),
        ),
        p=ElementRange(
            p_min,
            normalize_element_value(p_max, CONFIG["element_p_max"]),
        ),
        s=ElementRange(
            s_min,
            normalize_element_value(s_max, CONFIG["element_s_max"]),
        ),
        f_state=f_state,
        cl_state=cl_state,
        br_state=br_state,
        i_state=i_state,
    )


@app.function
def build_active_filters_dict(
    mass_filter_active: bool,
    mass_min_val: Optional[float],
    mass_max_val: Optional[float],
    year_filter_active: bool,
    year_start_val: Optional[int],
    year_end_val: Optional[int],
    formula_filters: Optional[FormulaFilters],
    smiles: Optional[str] = None,
    smiles_search_type: Optional[str] = None,
    smiles_threshold: Optional[float] = None,
) -> Dict[str, Any]:
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
        search_type = filters["chemical_structure"].get("search_type", "substructure")
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
    from urllib.parse import urlencode

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
        if c_min > 0:
            params["c_min"] = str(c_min)
        if c_max != CONFIG["element_c_max"]:
            params["c_max"] = str(c_max)
        if h_min > 0:
            params["h_min"] = str(h_min)
        if h_max != CONFIG["element_h_max"]:
            params["h_max"] = str(h_max)
        if n_min > 0:
            params["n_min"] = str(n_min)
        if n_max != CONFIG["element_n_max"]:
            params["n_max"] = str(n_max)
        if o_min > 0:
            params["o_min"] = str(o_min)
        if o_max != CONFIG["element_o_max"]:
            params["o_max"] = str(o_max)
        if p_min > 0:
            params["p_min"] = str(p_min)
        if p_max != CONFIG["element_p_max"]:
            params["p_max"] = str(p_max)
        if s_min > 0:
            params["s_min"] = str(s_min)
        if s_max != CONFIG["element_s_max"]:
            params["s_max"] = str(s_max)

        # Halogen states (only add non-default)
        if f_state != "allowed":
            params["f_state"] = f_state
        if cl_state != "allowed":
            params["cl_state"] = cl_state
        if br_state != "allowed":
            params["br_state"] = br_state
        if i_state != "allowed":
            params["i_state"] = i_state

    # Build URL
    if params:
        query_string = urlencode(params)
        return f"?{query_string}"
    else:
        return ""


@app.function
def compress_if_large(
    data: str, filename: str, threshold_bytes: int = None
) -> tuple[str, str, str]:
    """Compress data with gzip if it exceeds the threshold."""
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
    """Generic range filter for DataFrame columns."""
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
    """Check if a molecular formula matches the specified criteria."""
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
    """Apply molecular formula filters to the dataframe."""
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
    smiles: Optional[str] = None,
    search_mode: str = "taxon",
    smiles_search_type: str = "substructure",
    smiles_threshold: float = 0.8,
) -> pl.DataFrame:
    """
    Query Wikidata for compounds associated to taxa using multiple search strategies.

    Supports three search modes:
    1. Taxon-only: Find all compounds in a taxonomic group
    2. SMILES-only: Find compounds by chemical structure (SACHEM)
    3. Combined: Find structures within a specific taxonomic group
    """
    # Build query based on search mode
    if search_mode == "both" and smiles and qid:
        # Combined taxon + SMILES search
        query = build_smiles_taxon_query(
            smiles, qid, smiles_search_type, smiles_threshold
        )
    elif search_mode == "smiles" and smiles:
        # SMILES-only search
        if smiles_search_type == "similarity":
            query = build_smiles_similarity_query(smiles, smiles_threshold)
        else:  # Default to substructure
            query = build_smiles_substructure_query(smiles)
    elif qid == "*":
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
        "Reference title": row["ref_title"] or "-",
        "Reference DOI": create_link(f"https://doi.org/{doi}", doi)
        if doi
        else mo.Html("-"),
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
                f"(SMILES: {smiles_str[:50]}{'...' if len(smiles_str) > 50 else ''}) "
                f"within taxon {taxon_input} (Wikidata QID: {qid}). "
            )
        else:
            # SMILES-only search
            dataset_name = f"LOTUS Data - Chemical {search_type.title()} Search"
            description = (
                f"Chemical compounds matching {search_type} search "
                f"(SMILES: {smiles_str[:50]}{'...' if len(smiles_str) > 50 else ''}). "
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

    description += "Retrieved via LOTUS Wikidata Explorer with professional-grade chemical search capabilities (SACHEM/IDSM)."

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
            "license": "https://www.gnu.org/licenses/agpl-3.0.html",
            "applicationCategory": "Scientific Research Tool",
            "operatingSystem": "Platform Independent",
            "softwareRequirements": "Python 3.13+, Marimo",
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
                "description": "Free and Open Knowledge Base",
            },
            {
                "@type": "Organization",
                "name": "IDSM (Integrated Database of Small Molecules)",
                "url": "https://idsm.elixir-czech.cz/",
                "description": "SACHEM Chemical Search Service Provider",
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
            {
                "@type": "DataDownload",
                "encodingFormat": "text/turtle",
                "contentUrl": "data:text/turtle",
                "description": "RDF/Turtle format with semantic web annotations",
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
        "chemical_search_service": {
            "name": "SACHEM",
            "provider": "IDSM",
            "endpoint": CONFIG["idsm_endpoint"],
            "capabilities": ["substructure_search", "similarity_search"],
        },
    }

    # Add filters if any are active
    if filters:
        metadata["search_parameters"]["filters"] = filters

    return metadata


@app.function
def create_citation_text(taxon_input: str) -> str:
    """Generate citation text for the exported data."""
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"""
## 📖 How to Cite This Data

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
def export_to_rdf_turtle(df: pl.DataFrame, taxon_input: str, qid: str) -> str:
    """Export data to RDF Turtle format using rdflib following W3C standards."""
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
        This app is work in progress and may not work as expected in all deployments.
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
    # URL Query API section (left)
    url_api_section = mo.accordion(
        {
            "🔗 URL Query API": mo.md("""
            You can query this notebook via URL parameters! When running locally or accessing the published version, add query parameters to automatically execute searches.

            ### Available Parameters

            - `taxon` - Taxon name, QID, or **"*"** for all taxa
            - `smiles` - SMILES string for chemical structure search (optional)
            - `smiles_search_type` - "substructure" (default) or "similarity"
            - `smiles_threshold` - Similarity threshold (0.0-1.0, default: 0.8) - only used when `smiles_search_type=similarity`
            - **Combination behavior:**
              - Both `taxon` and `smiles`: Search for structure within that taxon
              - `smiles` only (taxon empty or "*"): Search across all compounds
              - `taxon` only (smiles empty): Search all compounds in taxon
            - `mass_filter=true` - Enable mass filter
            - `mass_min`, `mass_max` - Mass range in Daltons
            - `year_filter=true` - Enable year filter
            - `year_start`, `year_end` - Publication year range
            - `formula_filter=true` - Enable formula filter
            - `exact_formula` - Exact molecular formula (e.g., C15H10O5)
            - `c_min`, `c_max` - Carbon count range
            - `h_min`, `h_max` - Hydrogen count range
            - `n_min`, `n_max` - Nitrogen count range
            - `o_min`, `o_max` - Oxygen count range
            - `p_min`, `p_max` - Phosphorus count range
            - `s_min`, `s_max` - Sulfur count range
            - `f_state`, `cl_state`, `br_state`, `i_state` - Halogen states (allowed/required/excluded)

            ### Examples

            #### Search by structure within a taxon (combined)

            ```text
            ?taxon=Salix&smiles=CC(=O)Oc1ccccc1C(=O)O&smiles_search_type=substructure
            ```

            #### Search by structure across all compounds with similarity threshold

            ```text
            ?smiles=CC(=O)Oc1ccccc1C(=O)O&smiles_search_type=similarity&smiles_threshold=0.9
            ```

            #### Search by taxon name with mass filter

            ```text
            ?taxon=Swertia&mass_filter=true&mass_min=200&mass_max=600
            ```

            #### Search by QID with year and carbon range

            ```text
            ?taxon=Q157115&year_filter=true&year_start=2000&formula_filter=true&c_min=15&c_max=25
            ```

            #### Search excluding fluorine and requiring chlorine

            ```text
            ?taxon=Artemisia&formula_filter=true&f_state=excluded&cl_state=required
            ```

            #### Search all taxa with mass filter

            ```text
            ?taxon=*&mass_filter=true&mass_min=300&mass_max=500
            ```

            **Tip:** Copy the query parameters above and append them to your notebook URL.
            """)
        }
    )

    # Help & Documentation section (right)
    help_section = mo.accordion(
        {
            "❓ Help & Documentation": mo.md("""
            ### Quick Start Guide

            1. **Enter a taxon name** (e.g., "Artemisia annua") or Wikidata QID (e.g., "Q157115")
            2. **Optional:** Enter a SMILES string for structure-based search
               - Example: `CC(=O)Oc1ccccc1C(=O)O` (aspirin)
               - Example: `c1ccccc1` (benzene)
            3. **Search modes** (automatic based on input):
               - **Taxon + SMILES**: Find specific structures within a taxonomic group
               - **SMILES only**: Find structures across all compounds (leave taxon empty or use "*")
               - **Taxon only**: Find all compounds in a taxonomic group (leave SMILES empty)
            4. **SMILES search type**:
               - **Substructure**: Find compounds containing your structure (exact match)
               - **Similarity**: Find structurally similar compounds (uses Tanimoto coefficient)
            5. **Similarity threshold** (when using similarity search):
               - Adjust slider from 0.0 to 1.0 (default: 0.8)
               - Higher = more similar, fewer results
               - Lower = less similar, more results
            6. **Optional filters**: Mass range, publication year, molecular formula constraints
            7. **Click "🔍 Search Wikidata"** to retrieve data
            8. **Export**: Download results in CSV, JSON, RDF/Turtle, or with full metadata

            ### Features

            #### Search Capabilities

            **Taxonomic Search** 🔬
            - Search by scientific name (case-insensitive partial matching)
            - Direct search by Wikidata QID for precision (e.g., Q157115)
            - Wildcard search with **"*"** to query all taxa
            - **Smart disambiguation**: Automatically selects the taxon with most compound data when names overlap
            - Taxonomic hierarchy traversal (searches include all descendants)
            - Helpful suggestions for ambiguous or misspelled names

            **Chemical Structure Search** 🧪 (Powered by SACHEM/IDSM)
            - **Professional-grade** chemical search using SACHEM technology
            - **Substructure search**:
              - Graph-based matching (not text search)
              - Stereochemistry-aware
              - Respects chemical bond orders
              - Example use: Find all alkaloids, find phenol-containing compounds
            - **Similarity search**:
              - Fingerprint-based Tanimoto similarity
              - Adjustable threshold (0.0-1.0)
              - Example use: Find aspirin analogs, discover compound families
            - Can be combined with taxon for targeted searches
            - Example: "Find salicylates in Salix (willow) species"

            **Combined Search** 🔬 + 🧪
            - Search for specific chemical structures within taxonomic groups
            - Validate ethnobotanical knowledge
            - Discover chemotaxonomic patterns
            - Example: "Find quinoline alkaloids in Cinchona"

            #### Filtering Options

            **Mass Filter** ⚖️  
            Filter compounds by molecular mass (in Daltons)

            **Molecular Formula Filter** ⚛️  
            - Search by exact formula (e.g., C15H10O5)
            - Set element ranges (C, H, N, O, P, S)
            - Control halogen presence (F, Cl, Br, I):
              - *Allowed*: Can be present or absent
              - *Required*: Must be present
              - *Excluded*: Must not be present

            **Publication Year Filter** 🗓️  
            Filter by the year references were published

            #### Data Export

            - **CSV**: Spreadsheet-compatible format
            - **JSON**: Machine-readable structured data
            - **RDF/Turtle**: Semantic web format
              - Small datasets (≤{CONFIG["lazy_generation_threshold"]:,} rows): Generated automatically
              - Large datasets (>{CONFIG["lazy_generation_threshold"]:,} rows): Click "Generate RDF/Turtle" button to create on-demand
            - **Metadata**: Schema.org-compliant metadata with provenance
            - **Citation**: Proper citations for your publications

            **Note:** For large datasets (>{CONFIG["lazy_generation_threshold"]:,} rows), export generation is deferred for performance. Click the generation buttons when you're ready to create exports. Files are automatically compressed when >8MB.
            """)
        }
    )

    # Display side by side
    mo.hstack([url_api_section, help_section], gap=2, widths="equal")
    return


@app.cell
def _():
    # URL parameter detection and display
    _url_params_check = mo.query_params()

    # Display URL query info if parameters are present
    if _url_params_check and (
        "taxon" in _url_params_check or "smiles" in _url_params_check
    ):
        param_items = []

        # List known parameters in logical order
        known_params = [
            "taxon",
            "smiles",
            "smiles_search_type",
            "smiles_threshold",
            "mass_filter",
            "mass_min",
            "mass_max",
            "year_filter",
            "year_start",
            "year_end",
            "formula_filter",
            "exact_formula",
            "c_min",
            "c_max",
            "h_min",
            "h_max",
            "n_min",
            "n_max",
            "o_min",
            "o_max",
            "p_min",
            "p_max",
            "s_min",
            "s_max",
            "f_state",
            "cl_state",
            "br_state",
            "i_state",
        ]

        # Only show parameters that are actually present
        for key in known_params:
            if key in _url_params_check:
                value = _url_params_check.get(key)
                param_items.append(f"- **{key}**: `{value}`")

        if param_items:  # Only show if we found any parameters
            mo.callout(
                mo.md(f"""
                ### 🔗 URL Query Detected

                {chr(10).join(param_items)}

                The search will auto-execute with these parameters.
                """),
                kind="info",
            )
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
    state_smiles,
    state_smiles_search_type,
    state_smiles_threshold,
    state_taxon,
    state_year_end,
    state_year_filter,
    state_year_start,
):
    ## TAXON INPUT
    taxon_input = mo.ui.text(
        value=state_taxon,
        label="🔬 Taxon Name or Wikidata QID - Optional",
        placeholder="e.g., Artemisia annua, Cinchona, Q157115, or * for all taxa",
        full_width=True,
    )

    ## SMILES INPUT
    smiles_input = mo.ui.text(
        value=state_smiles,
        label="🧪 Chemical Structure (SMILES) - Optional",
        placeholder="e.g., c1ccccc1 (benzene), CC(=O)Oc1ccccc1C(=O)O (aspirin)",
        full_width=True,
    )

    smiles_search_type = mo.ui.dropdown(
        options=["substructure", "similarity"],
        value=state_smiles_search_type,
        label="🔍 Chemical Search Type",
        full_width=True,
    )

    smiles_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=state_smiles_threshold,
        label="🎯 Similarity Threshold",
        full_width=True,
    )

    ## MASS FILTERS
    mass_filter = mo.ui.checkbox(label="⚖️ Filter by mass", value=state_mass_filter)

    mass_min = mo.ui.number(
        value=state_mass_min,
        start=0,
        stop=CONFIG["mass_ui_max"],
        step=0.001,
        label="Min mass (Da)",
        full_width=True,
    )

    mass_max = mo.ui.number(
        value=state_mass_max,
        start=0,
        stop=CONFIG["mass_ui_max"],
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
        smiles_input,
        smiles_search_type,
        smiles_threshold,
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
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    filters_ui = [
        mo.md("## 🔍 Search Parameters"),
        taxon_input,
        smiles_input,
        smiles_search_type,
    ]

    # Show threshold slider only when similarity is selected
    if smiles_search_type.value == "similarity":
        filters_ui.append(smiles_threshold)

    filters_ui.extend(
        [
            mo.md("### Optional Filters"),
            mo.hstack([mass_filter], justify="start"),
            mo.hstack([mass_min, mass_max], gap=2, widths="equal")
            if mass_filter.value
            else mo.Html(""),
            mo.hstack([year_filter], justify="start"),
            mo.hstack([year_start, year_end], gap=2, widths="equal")
            if year_filter.value
            else mo.Html(""),
            mo.hstack([formula_filter], justify="start"),
        ]
    )

    if formula_filter.value:
        filters_ui.extend(
            [
                exact_formula,
                mo.md("**Element count ranges** (leave blank for no constraint)"),
                mo.hstack([c_min, c_max], gap=2, widths="equal"),
                mo.hstack([h_min, h_max], gap=2, widths="equal"),
                mo.hstack([n_min, n_max], gap=2, widths="equal"),
                mo.hstack([o_min, o_max], gap=2, widths="equal"),
                mo.hstack([p_min, p_max], gap=2, widths="equal"),
                mo.hstack([s_min, s_max], gap=2, widths="equal"),
                mo.md("**Halogen constraints**"),
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
    smiles_input,
    smiles_search_type,
    smiles_threshold,
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

        if use_smiles and use_taxon:
            # Both present - search by structure within taxon
            spinner_message = (
                f"🔎 Searching for SMILES '{smiles_str[:30]}...' in {taxon_input_str}"
            )
        elif use_smiles:
            # SMILES only
            spinner_message = f"🔎 Searching for SMILES: {smiles_str[:50]}..."
            qid = None
            taxon_warning = None
        else:
            # Taxon only
            if taxon_input_str == "*":
                spinner_message = "🔎 Searching all taxa ..."
            else:
                spinner_message = f"🔎 Searching for: {taxon_input_str}"

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
                                f"- Use a Wikidata QID directly (e.g., Q157115)"
                            ),
                            kind="warn",
                        ),
                    )

            try:
                y_start = year_start.value if year_filter.value else None
                y_end = year_end.value if year_filter.value else None
                m_min = mass_min.value if mass_filter.value else None
                m_max = mass_max.value if mass_filter.value else None

                # Build formula filters using factory function (DRY)
                formula_filt = None
                if formula_filter.value:
                    formula_filt = create_formula_filters(
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

                # Execute query based on what inputs are provided
                if use_smiles and use_taxon:
                    # Both SMILES and taxon - search for structure within taxon
                    results_df = query_wikidata(
                        qid=qid,
                        mass_min=m_min,
                        mass_max=m_max,
                        formula_filters=formula_filt,
                        smiles=smiles_str,
                        search_mode="both",
                        smiles_search_type=smiles_search_type.value,
                        smiles_threshold=smiles_threshold.value,
                    )
                elif use_smiles:
                    # SMILES only
                    results_df = query_wikidata(
                        qid="",
                        mass_min=m_min,
                        mass_max=m_max,
                        formula_filters=formula_filt,
                        smiles=smiles_str,
                        search_mode="smiles",
                        smiles_search_type=smiles_search_type.value,
                        smiles_threshold=smiles_threshold.value,
                    )
                else:
                    # Taxon only
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
def _(
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
    results_df,
    run_button,
    s_max,
    s_min,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    state_auto_run,
    taxon_input,
    taxon_warning,
    year_end,
    year_filter,
    year_start,
):
    # Display summary if either button was clicked or auto-run from URL
    if (not run_button.value and not state_auto_run) or results_df is None:
        summary_and_downloads = mo.Html("")
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
        summary_and_downloads = mo.vstack(parts) if len(parts) > 1 else parts[0]
    else:
        n_compounds = results_df.n_unique(subset=["structure"])
        n_taxa = results_df.n_unique(subset=["taxon"])
        n_refs = results_df.n_unique(subset=["reference"])
        n_entries = len(results_df)

        # Results header (on its own line)
        results_header = mo.md("## Results")

        # Taxon info
        if qid == "*":
            taxon_info = mo.md("**Search scope:** All taxa in LOTUS")
        else:
            taxon_info = mo.md(
                f"**Taxon:** {taxon_input.value} {create_wikidata_link(qid)}"
            )

        # Add SMILES search info if present
        search_info_parts = [taxon_info]
        if smiles_input.value and smiles_input.value.strip():
            _smiles_str = smiles_input.value.strip()
            search_type = smiles_search_type.value

            # Truncate long SMILES for display
            display_smiles = (
                _smiles_str if len(_smiles_str) <= 50 else f"{_smiles_str[:47]}..."
            )

            if search_type == "similarity":
                threshold_val = smiles_threshold.value
                smiles_info = mo.md(
                    f"**Chemical search:** {search_type.title()} "
                    f"(SMILES: `{display_smiles}`, "
                    f"Tanimoto threshold: **{threshold_val}**)"
                )
            else:
                smiles_info = mo.md(
                    f"**Chemical search:** {search_type.title()} "
                    f"(SMILES: `{display_smiles}`)"
                )
            search_info_parts.append(smiles_info)

        # Combine search info
        if len(search_info_parts) > 1:
            combined_search_info = mo.vstack(search_info_parts, gap=1)
        else:
            combined_search_info = search_info_parts[0]

        # Stats cards
        stats_cards = mo.hstack(
            [
                mo.stat(
                    value=f"{n_compounds:,}",
                    label=f"🧪 {pluralize('Compound', n_compounds)}",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{n_taxa:,}",
                    label=f"🌱 {pluralize('Taxon', n_taxa)}",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{n_refs:,}",
                    label=f"📚 {pluralize('Reference', n_refs)}",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{n_entries:,}",
                    label=f"📝 {pluralize('Entry', n_entries)}",
                    bordered=True,
                ),
            ],
            gap=2,
            justify="start",
            wrap=False,
        )

        # Search info and stats layout
        if len(search_info_parts) > 1:
            # Stack vertically when SMILES is present for better readability
            search_and_stats = mo.vstack(
                [combined_search_info, stats_cards],
                gap=2,
            )
        else:
            # Single line when no SMILES
            search_and_stats = mo.hstack(
                [combined_search_info, stats_cards],
                justify="space-between",
                align="start",
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
                **🔗 Shareable URL**

                Copy and append this to your notebook URL to share this exact search:
                ```
                {api_url}
                ```
                """
            )
            api_url_section = mo.accordion(
                {"🔁 Share this search": url_display},
                multiple=False,
            )
        else:
            api_url_section = mo.Html("")

        # Build summary section with all parts
        summary_parts = [results_header, search_and_stats]

        if api_url:
            summary_parts.append(api_url_section)

        if taxon_warning:
            summary_parts.append(mo.callout(taxon_warning, kind="warn"))

        summary_section = mo.vstack(summary_parts)

        # Combine summary (left) and downloads (right) side by side
        summary_and_downloads = mo.hstack(
            [summary_section, download_ui],
            justify="space-between",
            align="start",
            gap=4,
        )

    summary_and_downloads
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
    smiles_input,
    smiles_search_type,
    smiles_threshold,
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

        # Build filters for metadata using factory function (DRY)
        _formula_filt = None
        if formula_filter.value:
            _formula_filt = create_formula_filters(
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
                    "Reference title": row["ref_title"] or "-",
                    "Reference DOI": create_link(f"https://doi.org/{doi}", doi)
                    if (doi := row["ref_doi"])
                    else mo.Html("-"),
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
                    f"⚡ **Large Dataset Optimization**\n\n"
                    f"Your search returned **{total_rows:,} rows**. For optimal performance:\n"
                    f"- Displaying first **{CONFIG['table_row_limit']:,} rows** in table view\n"
                    f"- 2D structure images hidden (available in full download)\n"
                    f"- Use the Export View tab to see all data without images"
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
        else:
            csv_generate_button = None
            json_generate_button = None
            rdf_generate_button = None
            csv_generation_data = None
            json_generation_data = None
            rdf_generation_data = None

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
            [mo.md("### Download Data"), mo.hstack(buttons, gap=2, wrap=True)]
        )
        tables_ui = mo.vstack(
            [
                mo.md("### Browse Data"),
                display_note,
                mo.ui.tabs(
                    {
                        "🖼️ Display": display_table,
                        "📥 Export View": export_table,
                        "📖 Citation": mo.md(citation_text),
                        "🏷️ Metadata": mo.md(f"```json\n{metadata_json}\n```"),
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
                        f"✅ **CSV Ready** - {len(csv_generation_data['export_df']):,} entries"
                        + (
                            " (compressed)"
                            if _csv_mimetype == "application/gzip"
                            else ""
                        )
                    ),
                    kind="success",
                ),
                mo.download(
                    data=_csv_data,
                    filename=_csv_filename,
                    label="📥 Download CSV",
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
                        f"✅ **JSON Ready** - {len(json_generation_data['export_df']):,} entries"
                        + (
                            " (compressed)"
                            if _json_mimetype == "application/gzip"
                            else ""
                        )
                    ),
                    kind="success",
                ),
                mo.download(
                    data=_json_data,
                    filename=_json_filename,
                    label="📥 Download JSON",
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
                        f"✅ **RDF/Turtle Ready** - {len(rdf_generation_data['export_df']):,} entries"
                        + (
                            " (compressed)"
                            if _rdf_mimetype == "application/gzip"
                            else ""
                        )
                    ),
                    kind="success",
                ),
                mo.download(
                    data=_rdf_data,
                    filename=_rdf_filename,
                    label="📥 Download RDF/Turtle",
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
    url_auto_search = "taxon" in url_params or "smiles" in url_params

    # Get URL parameter values with defaults
    url_taxon = url_params.get("taxon", "Gentiana lutea")

    # SMILES search parameters
    url_smiles = url_params.get("smiles", "")
    url_smiles_search_type = url_params.get("smiles_search_type", "substructure")
    url_smiles_threshold = float(url_params.get("smiles_threshold", "0.8"))

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
        url_smiles,
        url_smiles_search_type,
        url_smiles_threshold,
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
    url_smiles,
    url_smiles_search_type,
    url_smiles_threshold,
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

        # SMILES search state
        state_smiles = url_smiles
        state_smiles_search_type = url_smiles_search_type
        state_smiles_threshold = url_smiles_threshold

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
        state_c_max = url_c_max if url_formula_filter else None
        state_h_min = url_h_min if url_formula_filter else None
        state_h_max = url_h_max if url_formula_filter else None
        state_n_min = url_n_min if url_formula_filter else None
        state_n_max = url_n_max if url_formula_filter else None
        state_o_min = url_o_min if url_formula_filter else None
        state_o_max = url_o_max if url_formula_filter else None
        state_p_min = url_p_min if url_formula_filter else None
        state_p_max = url_p_max if url_formula_filter else None
        state_s_min = url_s_min if url_formula_filter else None
        state_s_max = url_s_max if url_formula_filter else None
        state_f_state = url_f_state if url_formula_filter else "allowed"
        state_cl_state = url_cl_state if url_formula_filter else "allowed"
        state_br_state = url_br_state if url_formula_filter else "allowed"
        state_i_state = url_i_state if url_formula_filter else "allowed"

        state_auto_run = True

        mo.md(
            f"**Auto-executing search for:** {url_taxon if url_taxon else url_smiles}"
        )
    else:
        # Default states when no URL parameters
        state_taxon = "Gentiana lutea"
        state_smiles = ""
        state_smiles_search_type = "substructure"
        state_smiles_threshold = 0.8
        state_mass_filter = False
        state_mass_min = CONFIG["mass_default_min"]
        state_mass_max = CONFIG["mass_default_max"]
        state_year_filter = False
        state_year_start = CONFIG["year_default_start"]
        state_year_end = datetime.now().year
        state_formula_filter = False
        state_exact_formula = ""
        state_c_min = None
        state_c_max = None
        state_h_min = None
        state_h_max = None
        state_n_min = None
        state_n_max = None
        state_o_min = None
        state_o_max = None
        state_p_min = None
        state_p_max = None
        state_s_min = None
        state_s_max = None
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
        state_smiles,
        state_smiles_search_type,
        state_smiles_threshold,
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
    **Code:** <a href="https://github.com/cdk/depict" style="color:#339966;">CDK Depict</a> & <a href="https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py" style="color:#339966;">lotus_wikidata_explorer.py</a>  |  
    **License:** <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#006699;">CC0 1.0</a> for data & <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#006699;">AGPL-3.0</a> for code
    """
    )
    return


if __name__ == "__main__":
    app.run()
