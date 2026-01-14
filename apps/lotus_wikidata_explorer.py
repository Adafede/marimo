# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "great-tables==0.20.0",
#     "marimo",
#     "polars==1.37.1",
#     "rdflib==7.5.0",
# ]
# [tool.marimo.display]
# theme = "system"
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

__generated_with = "0.19.2"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import polars as pl
    import json
    import re
    import time
    import gzip
    import hashlib
    import sys
    import urllib.request
    import urllib.parse
    from dataclasses import dataclass, field
    from datetime import datetime
    from functools import lru_cache
    from great_tables import GT, html
    from rdflib import Graph, Namespace, Literal, URIRef, BNode
    from rdflib.namespace import RDF, RDFS, XSD, DCTERMS
    from typing import Optional, Dict, Any, Tuple, List, Mapping
    from urllib.parse import quote as url_quote

    # Patch urllib for Pyodide/WASM (browser) compatibility
    IS_PYODIDE = "pyodide" in sys.modules
    if IS_PYODIDE:
        import pyodide_http

        pyodide_http.patch_all()

    # ====================================================================
    # CENTRALIZED RDF NAMESPACES AND URLS
    # ====================================================================

    # RDF Namespaces wrapped in a dict to prevent marimo serialization issues
    # Access via RDF_NS["WD"], RDF_NS["SCHEMA"], etc.
    RDF_NS = {
        "WD": Namespace("http://www.wikidata.org/entity/"),
        "WDREF": Namespace("http://www.wikidata.org/reference/"),
        "WDS": Namespace("http://www.wikidata.org/entity/statement/"),
        "WDT": Namespace("http://www.wikidata.org/prop/direct/"),
        "P": Namespace("http://www.wikidata.org/prop/"),
        "PS": Namespace("http://www.wikidata.org/prop/statement/"),
        "PR": Namespace("http://www.wikidata.org/prop/reference/"),
        "PROV": Namespace("http://www.w3.org/ns/prov#"),
        "SCHEMA": Namespace("http://schema.org/"),
    }

    # URLs (constants)
    SCHOLIA_URL = "https://scholia.toolforge.org/"
    WIKIDATA_URL = "https://www.wikidata.org/"
    WIKIDATA_HTTP_URL = WIKIDATA_URL.replace("https://", "http://")
    WIKIDATA_ENTITY_URL = WIKIDATA_HTTP_URL + "entity/"
    WIKIDATA_STATEMENT_URL = WIKIDATA_ENTITY_URL + "statement/"
    WIKIDATA_WIKI_URL = WIKIDATA_URL + "wiki/"

    # ====================================================================
    # APPLICATION CONFIGURATION
    # All magic numbers centralized here for easy maintenance.
    # ====================================================================

    CONFIG = {
        "app_version": "0.0.1",
        "app_name": "LOTUS Wikidata Explorer",
        "app_url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
        "cdk_base": "https://www.simolecule.com/cdkdepict/depict/cow/svg",
        "sparql_endpoint": "https://qlever.dev/api/wikidata",
        "idsm_endpoint": "https://idsm.elixir-czech.cz/sparql/endpoint/",
        "max_retries": 3,
        "retry_backoff": 2,
        "query_timeout": 300,
        "table_row_limit": 5000,
        "lazy_generation_threshold": 500,
        "download_embed_threshold_bytes": 500_000,
        "color_hyperlink": "#3377c4",
        "color_wikidata_blue": "#006699",
        "color_wikidata_green": "#339966",
        "color_wikidata_red": "#990000",
        "page_size_default": 10,
        "page_size_export": 25,
        "year_range_start": 1700,
        "year_default_start": 1900,
        "mass_default_min": 0,
        "mass_default_max": 2000,
        "mass_ui_max": 10000,
        "element_c_max": 100,
        "element_h_max": 200,
        "element_n_max": 50,
        "element_o_max": 50,
        "element_p_max": 20,
        "element_s_max": 20,
        "default_similarity_threshold": 0.8,
    }

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

    SPARQL_PREFIXES = """
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>
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
    COMPOUND_SELECT_VARS = """
    ?compound ?compoundLabel ?compound_inchikey ?compound_smiles_conn
    ?compound_smiles_iso ?compound_mass ?compound_formula
    ?taxon_name ?taxon
    ?ref_qid ?ref_title ?ref_doi ?ref_date
    ?statement ?ref
    """

    COMPOUND_MINIMAL_VARS = """
    ?compound ?compoundLabel ?compound_inchikey ?compound_smiles_conn
    """

    COMPOUND_INTERIM_VARS = (
        COMPOUND_MINIMAL_VARS
        + """
    ?taxon ?taxon_name ?ref_qid ?statement ?ref
    """
    )

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


@app.class_definition
class SPARQLWrapper:
    """Simple SPARQL wrapper using urllib (works in both native Python and Pyodide/WASM)."""

    def __init__(
        self,
        sparql_endpoint: str = "https://qlever.dev/api/wikidata",
        timeout: int = 120,
    ):
        self.endpoint = sparql_endpoint
        self.timeout = timeout

    def query(self, query: str, response_format: str = "json"):
        """Execute a SPARQL query. Returns an object with .json() method."""
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = urllib.parse.urlencode({"query": query}).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")

        # Return an object with .json() method for consistency
        class Response:
            def __init__(self, body: str):
                self._body = body

            def json(self):
                return json.loads(self._body)

        return Response(body)


@app.class_definition
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


@app.class_definition
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
        if any(r.is_active() for r in [self.c, self.h, self.n, self.o, self.p, self.s]):
            return True
        if any(
            s != "allowed"
            for s in [self.f_state, self.cl_state, self.br_state, self.i_state]
        ):
            return True
        return False


@app.class_definition
@dataclass
class SearchParams:
    """Consolidated search parameters - replaces 30+ individual state variables."""

    # Core search
    taxon: str = "Gentiana lutea"
    smiles: str = ""
    smiles_search_type: str = "substructure"
    smiles_threshold: float = 0.8

    # Mass filter
    mass_filter: bool = False
    mass_min: float = field(default_factory=lambda: CONFIG["mass_default_min"])
    mass_max: float = field(default_factory=lambda: CONFIG["mass_default_max"])

    # Year filter
    year_filter: bool = False
    year_start: int = field(default_factory=lambda: CONFIG["year_default_start"])
    year_end: int = field(default_factory=lambda: datetime.now().year)

    # Formula filter
    formula_filter: bool = False
    exact_formula: str = ""
    c_min: Optional[int] = None
    c_max: Optional[int] = None
    h_min: Optional[int] = None
    h_max: Optional[int] = None
    n_min: Optional[int] = None
    n_max: Optional[int] = None
    o_min: Optional[int] = None
    o_max: Optional[int] = None
    p_min: Optional[int] = None
    p_max: Optional[int] = None
    s_min: Optional[int] = None
    s_max: Optional[int] = None
    f_state: str = "allowed"
    cl_state: str = "allowed"
    br_state: str = "allowed"
    i_state: str = "allowed"

    # Auto-run flag
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
                int(params.get(f"{e}_max", CONFIG[f"element_{e}_max"])) if ff else None
            )

        return cls(
            taxon=params.get("taxon", "Gentiana lutea"),
            smiles=params.get("smiles", ""),
            smiles_search_type=params.get("smiles_search_type", "substructure"),
            smiles_threshold=float(params.get("smiles_threshold", "0.8")),
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

    def to_formula_filters(self) -> Optional[FormulaFilters]:
        """Convert to FormulaFilters if formula filter is active."""
        if not self.formula_filter:
            return None
        elem_args = {}
        for e in ("c", "h", "n", "o", "p", "s"):
            min_v, max_v = getattr(self, f"{e}_min"), getattr(self, f"{e}_max")
            elem_args[f"{e}_min"] = min_v or 0
            elem_args[f"{e}_max"] = (
                max_v if max_v is not None else CONFIG[f"element_{e}_max"]
            )
        return create_formula_filters(
            exact_formula=self.exact_formula,
            **elem_args,
            f_state=self.f_state,
            cl_state=self.cl_state,
            br_state=self.br_state,
            i_state=self.i_state,
        )


@app.function
def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """Validate SMILES string for common issues."""
    # Empty is valid (means no SMILES search)
    if not smiles or not smiles.strip():
        return True, None

    smiles = smiles.strip()

    # Length validation
    if len(smiles) < 1:
        return False, "SMILES string is empty after trimming whitespace"
    if len(smiles) > 10000:
        return False, (
            f"SMILES string is too long ({len(smiles):,} characters). "
            f"Maximum allowed: 10,000 characters. "
            f"Please use a simpler structure or substructure."
        )

    # Check for null bytes or dangerous control characters
    if "\x00" in smiles:
        return (
            False,
            "SMILES contains null bytes (\\x00) which are not allowed",
        )

    invalid_chars = [c for c in smiles if ord(c) < 32 and c not in "\t\n\r"]
    if invalid_chars:
        chars_display = ", ".join(f"\\x{ord(c):02x}" for c in invalid_chars[:3])
        return False, (
            f"SMILES contains invalid control characters: {chars_display}. "
            f"Only standard ASCII printable characters are allowed."
        )

    # Basic sanity check: should contain at least one atom symbol
    # Common atom symbols in SMILES: C, N, O, S, P, F, Cl, Br, I, B, c (aromatic), etc.
    if not any(c in smiles for c in "CNOSPFIBcnops"):
        return False, (
            "SMILES appears to be missing atom symbols. "
            "A valid SMILES must contain at least one element (C, N, O, etc.). "
            "Example: 'c1ccccc1' for benzene"
        )

    return True, None


@app.function
def escape_smiles_for_sparql(smiles: str) -> str:
    """
    Escape SMILES string for safe use in SPARQL queries.

    SMILES strings can contain backslashes (e.g., /C=C\3/) which are escape
    characters in SPARQL string literals and must be escaped.
    """
    if not smiles:
        return smiles

    # Validate SMILES
    is_valid, error_msg = validate_smiles(smiles)
    if not is_valid:
        raise ValueError(f"Invalid SMILES: {error_msg}")

    # Escape backslashes by doubling them
    return smiles.replace("\\", "\\\\")


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
def build_base_sachem_query(
    smiles: str,
    search_type: str = "substructure",
    threshold: float = 0.8,
    include_taxon_filter: bool = False,
    taxon_qid: Optional[str] = None,
) -> str:
    """Build base SACHEM chemical search query."""
    escaped_smiles = escape_smiles_for_sparql(smiles)

    # Build SACHEM service clause based on search type
    if search_type == "similarity":
        sachem_clause = f"""
        SERVICE idsm:wikidata {{
            VALUES ?QUERY_SMILES {{ "{escaped_smiles}" }}
            VALUES ?CUTOFF {{ "{threshold}"^^xsd:double }}
            ?compound sachem:similarCompoundSearch[
            sachem:query ?QUERY_SMILES;
            sachem:cutoff ?CUTOFF
            ].
        }}
        """
    else:
        # substructure
        sachem_clause = f"""
        SERVICE idsm:wikidata {{
            VALUES ?SUBSTRUCTURE {{ "{escaped_smiles}" }}
            ?compound sachem:substructureSearch [
                sachem:query ?SUBSTRUCTURE
            ].
        }}
        """

    # Build taxon filter if requested
    taxon_filter = ""
    if include_taxon_filter and taxon_qid:
        taxon_filter = f"""
        {TAXON_REFERENCE_ASSOCIATION}
        ?taxon (wdt:P171*) wd:{taxon_qid}
        {REFERENCE_METADATA_OPTIONAL}
        """
    else:
        taxon_filter = f"""
        {TAXONOMIC_REFERENCE_OPTIONAL}
        """

    # Construct full query
    return f"""
    {SPARQL_PREFIXES}{SACHEM_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
        {{
            SELECT {COMPOUND_MINIMAL_VARS} WHERE {{
                {sachem_clause}
                {COMPOUND_IDENTIFIERS}
            }}
        }}
        {taxon_filter}
        {COMPOUND_PROPERTIES_OPTIONAL}
    }}
    """


@app.function
def build_compounds_query(qid: str) -> str:
    """Build SPARQL query to find compounds in a specific taxon and its descendants."""
    return f"""
    {SPARQL_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
      {{
        SELECT {COMPOUND_INTERIM_VARS} WHERE {{
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
    return f"""
    {SPARQL_PREFIXES}
    SELECT {COMPOUND_SELECT_VARS} WHERE {{
        {{
            SELECT {COMPOUND_INTERIM_VARS} WHERE {{
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
    """Execute SPARQL query with retry logic. Works in both native Python and Pyodide/WASM."""
    if not query or not query.strip():
        raise ValueError("SPARQL query cannot be empty")

    def _wait(a):
        time.sleep(CONFIG["retry_backoff"] * (2**a))

    # Get wrapper once before retry loop
    sparql = SPARQLWrapper()

    for attempt in range(max_retries):
        try:
            result = sparql.query(query, response_format="json").json()
            if not isinstance(result, dict) or "results" not in result:
                raise ValueError("Invalid SPARQL response format")
            return result
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                raise ValueError(f"❌ Invalid JSON: {e}") from e
            _wait(attempt)
        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)

            # Handle timeout errors (both httpx and urllib)
            if "timeout" in error_name.lower() or "timeout" in error_msg.lower():
                if attempt == max_retries - 1:
                    raise TimeoutError(
                        f"⏱️ Query timed out after {max_retries} attempts. Try adding filters."
                    ) from e
                _wait(attempt)
            # Handle HTTP errors
            elif "http" in error_name.lower() or "urlerror" in error_name.lower():
                if attempt == max_retries - 1:
                    raise ConnectionError(f"🌐 HTTP error: {error_msg[:200]}") from e
                _wait(attempt)
            # Handle network errors
            elif "network" in error_name.lower() or "connection" in error_name.lower():
                if attempt == max_retries - 1:
                    raise ConnectionError(f"🌐 Network error: {e}") from e
                _wait(attempt)
            # Other errors
            else:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"❌ {error_name}: {e}") from e
                _wait(attempt)
    raise RuntimeError("Max retries exceeded")


@app.function
@lru_cache(maxsize=512)
def extract_qid(url: Optional[str]) -> Optional[str]:
    """Extract QID from Wikidata entity URL. Cached for performance."""
    if url is None:
        return None
    return url.replace(WIKIDATA_ENTITY_URL, "")


@app.function
@lru_cache(maxsize=1024)
def create_structure_image_url(smiles: Optional[str]) -> Optional[str]:
    if smiles is None:
        return None
    encoded_smiles = url_quote(smiles)
    return f"{CONFIG['cdk_base']}?smi={encoded_smiles}&annotate=cip"


@app.function
def create_lazy_structure_image(smiles: Optional[str]) -> mo.Html:
    """Create a lazy-loading structure image to reduce memory usage on mobile devices."""
    if smiles is None:
        return mo.Html("")
    url = create_structure_image_url(smiles)
    if url is None:
        return mo.Html("")
    # Use loading="lazy" and decoding="async" for better mobile performance
    return mo.Html(
        f'<img src="{url}" loading="lazy" decoding="async" '
        f'alt="Structure" style="max-width:200px;max-height:150px;" />'
    )


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
    {SPARQL_PREFIXES}

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
    {SPARQL_PREFIXES}

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
def resolve_ambiguous_matches(
    matches: List[Tuple[str, str]], is_exact: bool
) -> Tuple[str, mo.Html]:
    """Helper to resolve ambiguous taxon matches by connectivity. Returns (selected_qid, warning_html)."""
    qids = [qid for qid, _ in matches[:5]]  # Limit to 5 for performance

    # Query connectivity to find the most connected taxon
    connectivity_results = execute_sparql(build_taxon_connectivity_query(qids))
    connectivity_map = {
        extract_qid(get_binding_value(b, "taxon")): int(
            get_binding_value(b, "compound_count", "0")
        )
        for b in connectivity_results.get("results", {}).get("bindings", [])
    }

    # Sort by connectivity (descending)
    sorted_matches = sorted(
        matches[:5], key=lambda x: connectivity_map.get(x[0], 0), reverse=True
    )

    # Get details for display
    details_results = execute_sparql(build_taxon_details_query(qids))
    details_map = {
        extract_qid(get_binding_value(b, "taxon")): (
            get_binding_value(b, "taxonLabel"),
            get_binding_value(b, "taxonDescription"),
            get_binding_value(b, "taxon_parentLabel"),
        )
        for b in details_results.get("results", {}).get("bindings", [])
    }

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

    selected_qid = sorted_matches[0][0]
    return selected_qid, create_taxon_warning_html(
        matches_with_details, selected_qid, is_exact=is_exact
    )


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
        link = f'<a href="{SCHOLIA_URL}{qid}" target="_blank" rel="noopener noreferrer" style="color: {CONFIG["color_hyperlink"]}; text-decoration: none; border-bottom: 1px solid transparent; font-weight: bold;">{qid}</a>'

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
def resolve_taxon_to_qid(
    taxon_input: str,
) -> Tuple[Optional[str], Optional[mo.Html]]:
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
def get_binding_value(binding: Dict[str, Any], key: str, default: str = "") -> str:
    return binding.get(key, {}).get("value", default)


@app.function
def pluralize(singular: str, count: int) -> str:
    """Return singular or plural form based on count with special cases."""
    return singular if count == 1 else PLURAL_MAP.get(singular, f"{singular}s")


@app.function
def serialize_element_range(
    element_range: ElementRange,
) -> Optional[Dict[str, int]]:
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
    element_attrs = [
        filters.c,
        filters.h,
        filters.n,
        filters.o,
        filters.p,
        filters.s,
    ]
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
    elem_vals = {
        "c": (c_min, c_max),
        "h": (h_min, h_max),
        "n": (n_min, n_max),
        "o": (o_min, o_max),
        "p": (p_min, p_max),
        "s": (s_min, s_max),
    }
    ranges = {
        k: ElementRange(mn, normalize_element_value(mx, CONFIG[f"element_{k}_max"]))
        for k, (mn, mx) in elem_vals.items()
    }
    return FormulaFilters(
        exact_formula=exact_formula.strip()
        if exact_formula and exact_formula.strip()
        else None,
        **ranges,
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

        # Element ranges (only add non-default values) - use loop to reduce repetition
        element_vals = [
            ("c", c_min, c_max, "element_c_max"),
            ("h", h_min, h_max, "element_h_max"),
            ("n", n_min, n_max, "element_n_max"),
            ("o", o_min, o_max, "element_o_max"),
            ("p", p_min, p_max, "element_p_max"),
            ("s", s_min, s_max, "element_s_max"),
        ]
        for elem, min_val, max_val, max_key in element_vals:
            if min_val > 0:
                params[f"{elem}_min"] = str(min_val)
            if max_val != CONFIG[max_key]:
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
def create_download_button(
    data: str, filename: str, label: str, base_mimetype: str
) -> mo.download:
    """Create a download button with automatic compression for large files."""
    compressed_data, final_filename, final_mimetype = compress_if_large(data, filename)

    # Add compression indicator to label
    display_label = label + (
        " (gzipped)" if final_mimetype == "application/gzip" else ""
    )

    return mo.download(
        data=compressed_data,
        filename=final_filename,
        label=display_label,
        mimetype=final_mimetype if final_mimetype else base_mimetype,
    )


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
        df,
        "pub_date",
        year_start,
        year_end,
        extract_func=lambda col: col.dt.year(),
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
    # Input validation
    if search_mode not in ("taxon", "smiles", "combined"):
        raise ValueError(
            f"Invalid search_mode: '{search_mode}'. "
            f"Must be one of: 'taxon', 'smiles', 'combined'"
        )

    if smiles_search_type not in ("substructure", "similarity"):
        raise ValueError(
            f"Invalid smiles_search_type: '{smiles_search_type}'. "
            f"Must be one of: 'substructure', 'similarity'"
        )

    if not (0.0 <= smiles_threshold <= 1.0):
        raise ValueError(
            f"Invalid smiles_threshold: {smiles_threshold}. "
            f"Must be between 0.0 and 1.0"
        )

    if year_start is not None and year_end is not None and year_start > year_end:
        raise ValueError(
            f"Invalid year range: start ({year_start}) > end ({year_end}). "
            f"Start year must be <= end year."
        )

    if mass_min is not None and mass_max is not None and mass_min > mass_max:
        raise ValueError(
            f"Invalid mass range: min ({mass_min}) > max ({mass_max}). "
            f"Minimum mass must be <= maximum mass."
        )

    # Build query based on search mode
    if search_mode == "combined" and smiles and qid:
        query = build_base_sachem_query(
            smiles, smiles_search_type, smiles_threshold, True, qid
        )
    elif search_mode == "smiles" and smiles:
        query = build_base_sachem_query(smiles, smiles_search_type, smiles_threshold)
    elif qid == "*" or qid is None:
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
            "compound": get_binding_value(b, "compound"),
            "name": get_binding_value(b, "compoundLabel"),
            "inchikey": get_binding_value(b, "compound_inchikey"),
            "smiles": get_binding_value(b, "compound_smiles_iso")
            or get_binding_value(b, "compound_smiles_conn"),
            "taxon_name": get_binding_value(b, "taxon_name"),
            "taxon": get_binding_value(b, "taxon"),
            "ref_title": get_binding_value(b, "ref_title"),
            "ref_doi": (
                doi.split("doi.org/")[-1]
                if isinstance(doi := get_binding_value(b, "ref_doi"), str)
                and doi.startswith("http")
                else doi
            ),
            "reference": get_binding_value(b, "ref_qid"),
            "pub_date": get_binding_value(b, "ref_date", None),
            "mass": float(mass_raw)
            if (mass_raw := get_binding_value(b, "compound_mass", None))
            else None,
            "mf": get_binding_value(b, "compound_formula"),
            "statement": get_binding_value(b, "statement"),
            "ref": get_binding_value(b, "ref"),
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
    return df.unique(subset=["compound", "taxon", "reference"], keep="first").sort(
        "name"
    )


@app.function
def html_structure_image(smiles: str) -> str:
    """Return HTML img tag for 2D structure."""
    if not smiles:
        return ""
    img_url = f"{CONFIG['cdk_base']}?smi={url_quote(smiles)}&annotate=cip"
    return f'<img src="{img_url}" loading="lazy" style="max-width:150px;max-height:100px;border-radius:8px;"/>'


@app.function
def html_doi_link(doi: str) -> str:
    """Return HTML link for DOI."""
    if not doi:
        return ""
    return f'<a href="https://doi.org/{doi}" target="_blank" style="color:{CONFIG["color_hyperlink"]};">{doi}</a>'


@app.function
def html_qid_link(url: str, color: str) -> str:
    """Return HTML link for Wikidata QID."""
    if not url:
        return ""
    qid = url.replace(WIKIDATA_ENTITY_URL, "")
    return (
        f'<a href="{SCHOLIA_URL}{qid}" target="_blank" style="color:{color};">{qid}</a>'
    )


@app.function
def html_statement_link(url: str) -> str:
    """Return HTML link for statement."""
    if not url:
        return ""
    statement_id = url.split("/")[-1] if url else ""
    return f'<a href="{url}" target="_blank" style="color:{CONFIG["color_hyperlink"]};">{statement_id}</a>'


@app.function
def build_display_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Build display DataFrame with HTML-formatted columns.

    Returns a DataFrame with renamed columns and HTML strings for links/images.
    This can be used for both WASM (plain HTML) and native (mo.ui.table).
    """
    return df.select(
        [
            pl.col("smiles")
            .map_elements(html_structure_image, return_dtype=pl.String)
            .alias("Compound Depiction"),
            pl.col("name").alias("Compound Name"),
            pl.col("smiles").alias("Compound SMILES"),
            pl.col("inchikey").alias("Compound InChIKey"),
            pl.col("mf").alias("Compound Molecular Formula"),
            pl.col("mass").alias("Compound Mass"),
            pl.col("taxon_name").alias("Taxon Name"),
            pl.col("ref_title").alias("Reference Title"),
            pl.col("pub_date").alias("Reference Date"),
            pl.col("ref_doi")
            .map_elements(html_doi_link, return_dtype=pl.String)
            .alias("Reference DOI"),
            pl.col("compound")
            .map_elements(
                lambda x: html_qid_link(x, CONFIG["color_wikidata_red"]),
                return_dtype=pl.String,
            )
            .alias("Compound QID"),
            pl.col("taxon")
            .map_elements(
                lambda x: html_qid_link(x, CONFIG["color_wikidata_green"]),
                return_dtype=pl.String,
            )
            .alias("Taxon QID"),
            pl.col("reference")
            .map_elements(
                lambda x: html_qid_link(x, CONFIG["color_wikidata_blue"]),
                return_dtype=pl.String,
            )
            .alias("Reference QID"),
            pl.col("statement")
            .map_elements(html_statement_link, return_dtype=pl.String)
            .alias("Statement"),
        ]
    )


@app.function
def wrap_html(html_str: str) -> mo.Html:
    """Wrap HTML string in mo.Html for mo.ui.table."""
    return mo.Html(html_str) if html_str else mo.Html("")


@app.function
def wrap_image(html_str: str):
    """Wrap image HTML in mo.image for mo.ui.table, or return mo.Html as fallback."""
    if not html_str:
        return mo.Html("")
    # Extract src from img tag and use mo.image
    import re

    match = re.search(r'src="([^"]+)"', html_str)
    if match:
        return mo.image(src=match.group(1), width=150, height=100, rounded=True)
    return mo.Html(html_str)


@app.function
def prepare_export_dataframe(
    df: pl.DataFrame,
    include_rdf_ref: bool = False,
) -> pl.DataFrame:
    """
    Prepare dataframe for export with cleaned QIDs and selected columns.

    Args:
        df: Input dataframe (not mutated).
        include_rdf_ref: If True, include ref URI for RDF export.
                         Statement is always included if present.
    """
    df_with_qids: pl.DataFrame = df.with_columns(
        [
            pl.col("compound")
            .str.replace(WIKIDATA_ENTITY_URL, "", literal=True)
            .alias("compound_qid"),
            pl.col("taxon")
            .str.replace(WIKIDATA_ENTITY_URL, "", literal=True)
            .alias("taxon_qid"),
            pl.col("reference")
            .str.replace(WIKIDATA_ENTITY_URL, "", literal=True)
            .alias("reference_qid"),
            pl.col("statement")
            .str.replace(WIKIDATA_STATEMENT_URL, "", literal=True)
            .alias("statement_id"),
        ]
    )

    select_cols: list[pl.Expr | str] = [
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

    # Always include statement (for display + RDF)
    if "statement" in df_with_qids.columns:
        select_cols.append("statement_id")

    # Only include ref URI for RDF export
    if include_rdf_ref and "ref" in df_with_qids.columns:
        select_cols.append("ref")

    return df_with_qids.select(select_cols)


@app.function
def create_export_metadata(
    df: pl.DataFrame,
    taxon_input: str,
    qid: str,
    filters: Dict[str, Any],
    query_hash: Optional[str] = None,
    result_hash: Optional[str] = None,
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
            {"@type": "Organization", "name": n, "url": u}
            for n, u in [
                ("LOTUS Initiative", WIKIDATA_WIKI_URL + "Q104225190"),
                ("Wikidata", WIKIDATA_URL),
                ("IDSM", "https://idsm.elixir-czech.cz/"),
            ]
        ],
        "citation": [
            {
                "@type": "ScholarlyArticle",
                "name": "LOTUS initiative",
                "identifier": "https://doi.org/10.7554/eLife.70780",
            }
        ],
        "distribution": [
            {
                "@type": "DataDownload",
                "encodingFormat": f,
                "contentUrl": f"data:{f}",
            }
            for f in ["text/csv", "application/json", "text/turtle"]
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
        "search_parameters": {"taxon": taxon_input, "taxon_qid": qid},
        "sparql_endpoint": CONFIG["sparql_endpoint"],
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
def compute_provenance_hashes(
    qid: Optional[str],
    taxon_input: Optional[str],
    filters: Optional[Dict[str, Any]],
    df: pl.DataFrame,
) -> Tuple[str, str]:
    """
    Compute query and result hashes for provenance tracking.

    Returns:
        Tuple of (query_hash, result_hash) where:
        - query_hash: based on search parameters (what was asked)
        - result_hash: based on compound identifiers (what was found)
    """
    # Query hash
    query_components = [qid or "", taxon_input or ""]
    if filters:
        query_components.append(json.dumps(filters, sort_keys=True))
    query_hash = hashlib.sha256("|".join(query_components).encode("utf-8")).hexdigest()

    # Result hash - extract compound QIDs efficiently
    compound_col = "compound_qid" if "compound_qid" in df.columns else "compound"
    if compound_col in df.columns:
        compound_ids = sorted(
            df.select(
                pl.col(compound_col).str.replace(WIKIDATA_ENTITY_URL, "", literal=True)
            )
            .to_series()
            .drop_nulls()
            .unique()
            .to_list()
        )
    else:
        compound_ids = []
    result_hash = hashlib.sha256("|".join(compound_ids).encode("utf-8")).hexdigest()

    return query_hash, result_hash


@app.function
def create_dataset_uri(
    qid: str,
    taxon_input: str,
    filters: Optional[Dict[str, Any]],
    df: pl.DataFrame,
) -> Tuple[URIRef, str, str]:
    """
    Create dataset URI based on result content for reproducibility.

    NOTE: LOTUS data is hosted on Wikidata (https://www.wikidata.org/wiki/Q104225190).
    There is no separate LOTUS namespace - data is stored as regular Wikidata entities.
    This export creates a virtual dataset URI for the query result using a
    content-addressable URN based ONLY on what was found (not what was asked).
    The query hash is returned separately for metadata storage.
    """
    query_hash, result_hash = compute_provenance_hashes(qid, taxon_input, filters, df)

    # Create a content-addressable URI using URN scheme with ONLY result hash
    # Format: urn:hash:sha256:RESULT_HASH
    # This identifies the dataset by its content, not by how it was obtained
    dataset_uri = URIRef(f"urn:hash:sha256:{result_hash}")

    return dataset_uri, query_hash, result_hash


@app.function
def build_dataset_description(
    taxon_input: str, filters: Optional[Dict[str, Any]]
) -> Tuple[str, str]:
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
    SCHEMA = RDF_NS["SCHEMA"]
    WD = RDF_NS["WD"]

    # Dataset type and basic metadata
    g.add((dataset_uri, RDF.type, SCHEMA.Dataset))
    g.add((dataset_uri, SCHEMA.name, Literal(dataset_name, datatype=XSD.string)))
    g.add(
        (
            dataset_uri,
            SCHEMA.description,
            Literal(dataset_desc, datatype=XSD.string),
        )
    )

    # License and provenance - CC0 from Wikidata/LOTUS
    g.add(
        (
            dataset_uri,
            SCHEMA.license,
            URIRef("https://creativecommons.org/publicdomain/zero/1.0/"),
        )
    )
    g.add((dataset_uri, SCHEMA.provider, URIRef(CONFIG["app_url"])))
    g.add((dataset_uri, DCTERMS.source, URIRef(WIKIDATA_URL)))

    # Dataset statistics and versioning
    g.add(
        (
            dataset_uri,
            SCHEMA.numberOfRecords,
            Literal(df_len, datatype=XSD.integer),
        )
    )
    g.add(
        (
            dataset_uri,
            SCHEMA.version,
            Literal(CONFIG["app_version"], datatype=XSD.string),
        )
    )

    # Reference to LOTUS Initiative (Q104225190) as the source project
    g.add(
        (
            dataset_uri,
            SCHEMA.isBasedOn,
            URIRef(WIKIDATA_WIKI_URL + "Q104225190"),
        )
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
        )
    )
    # Result hash is implicit in the dataset URI itself (urn:hash:sha256:RESULT_HASH)
    # but we also add it explicitly for clarity
    g.add(
        (
            dataset_uri,
            DCTERMS.identifier,
            Literal(f"sha256:{result_hash}", datatype=XSD.string),
        )
    )


@app.function
def add_optional_literal(
    g: Graph,
    subject: URIRef,
    predicate: URIRef,
    value: Any,
    datatype=XSD.string,
) -> None:
    """Add optional literal triple to graph if value exists (DRY helper)."""
    if value is not None and value != "":
        g.add((subject, predicate, Literal(value, datatype=datatype)))


@app.function
def add_compound_triples(
    g: Graph,
    row: Dict[str, Any],
    dataset_uri: URIRef,
    processed_taxa: set,
    processed_refs: set,
) -> None:
    """Add all triples for a single compound using Wikidata's full RDF structure."""
    WD = RDF_NS["WD"]
    WDT = RDF_NS["WDT"]
    P = RDF_NS["P"]
    PS = RDF_NS["PS"]
    PR = RDF_NS["PR"]
    PROV = RDF_NS["PROV"]
    SCHEMA = RDF_NS["SCHEMA"]

    compound_qid = row.get("compound_qid", "")
    if not compound_qid:
        return

    compound_uri = WD[compound_qid]

    # Link compound to dataset
    g.add((dataset_uri, SCHEMA.hasPart, compound_uri))

    # Compound identifiers using Wikidata properties (direct properties)
    add_optional_literal(
        g, compound_uri, WDT.P235, row.get("compound_inchikey")
    )  # InChIKey
    add_optional_literal(
        g, compound_uri, WDT.P233, row.get("compound_smiles")
    )  # Canonical SMILES
    add_optional_literal(
        g, compound_uri, WDT.P274, row.get("molecular_formula")
    )  # Molecular formula

    # Mass (P2067)
    if row.get("compound_mass") is not None:
        add_optional_literal(
            g, compound_uri, WDT.P2067, row["compound_mass"], XSD.float
        )

    # Compound label
    add_optional_literal(g, compound_uri, RDFS.label, row.get("compound_name"))

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
        g.add((compound_uri, P.P703, statement_node))  # compound has a P703 statement
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
                add_optional_literal(g, ref_uri, WDT.P1476, row.get("reference_title"))
                add_optional_literal(g, ref_uri, RDFS.label, row.get("reference_title"))

                # P356: DOI
                if row.get("reference_doi"):
                    add_optional_literal(g, ref_uri, WDT.P356, row.get("reference_doi"))

                # P577: publication date
                if row.get("reference_date"):
                    add_optional_literal(
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
            add_optional_literal(g, taxon_uri, WDT.P225, row.get("taxon_name"))
            add_optional_literal(g, taxon_uri, RDFS.label, row.get("taxon_name"))
            processed_taxa.add(taxon_qid)


@app.function
def export_to_rdf_turtle(
    df: pl.DataFrame,
    taxon_input: str,
    qid: str,
    filters: Optional[Dict[str, Any]] = None,
) -> str:
    """Export data to RDF Turtle format using Wikidata's full RDF structure."""
    # Initialize graph
    g = Graph()

    # Bind namespaces from RDF_NS
    g.bind("wd", RDF_NS["WD"])
    g.bind("wdref", RDF_NS["WDREF"])
    g.bind("wds", RDF_NS["WDS"])
    g.bind("wdt", RDF_NS["WDT"])
    g.bind("p", RDF_NS["P"])
    g.bind("ps", RDF_NS["PS"])
    g.bind("pr", RDF_NS["PR"])
    g.bind("prov", RDF_NS["PROV"])
    g.bind("schema", RDF_NS["SCHEMA"])
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("dcterms", DCTERMS)

    # Create dataset URI with provenance hashes
    dataset_uri, query_hash, result_hash = create_dataset_uri(
        qid, taxon_input, filters, df
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

    # Add compound data
    for row in df.iter_rows(named=True):
        add_compound_triples(g, row, dataset_uri, processed_taxa, processed_refs)

    # Serialize to Turtle format
    return g.serialize(format="turtle")


@app.cell
def md_title():
    mo.md("""
    # 🌐 LOTUS Wikidata Explorer
    """)
    return


@app.cell
def ui_disclaimer():
    mo.callout(
        mo.md("""
        **Work in progress** - May not work in all deployments.  
        **Recommended:** `uvx marimo run https://raw.githubusercontent.com/Adafede/marimo/refs/heads/main/apps/lotus_wikidata_explorer.py`
        """),
        kind="info",
    )
    return


@app.cell
def ui_url_api():
    # URL Query API section (left)
    url_api_section = mo.accordion(
        {
            "🔗 URL Query API": mo.md("""
            Query via URL parameters. Add to notebook URL to auto-execute searches.

            **Key Parameters:**
            - `taxon` - Name, QID, or "*" for all
            - `smiles` - Chemical structure
            - `smiles_search_type` - "substructure" or "similarity"
            - `smiles_threshold` - 0.0-1.0 (for similarity)
            - `mass_filter=true`, `mass_min`, `mass_max`
            - `year_filter=true`, `year_start`, `year_end`
            - `formula_filter=true`, `exact_formula`
            - Element ranges: `c_min`, `c_max`, `h_min`, `h_max`, etc.
            - Halogen states: `f_state`, `cl_state`, `br_state`, `i_state`

            **Examples:**
            ```
            ?taxon=Salix&smiles=CC(=O)Oc1ccccc1C(=O)O
            ?smiles=c1ccccc1&smiles_search_type=similarity&smiles_threshold=0.45
            ?taxon=*&mass_filter=true&mass_min=300&mass_max=500
            ```
            """)
        }
    )

    # Help & Documentation section (right)
    help_section = mo.accordion(
        {
            "❓ Help": mo.md("""
            **Quick Start:** Enter taxon (or "*") → Add SMILES (optional) → Apply filters → Search

            **Search Modes:**
            - **Taxon only**: All compounds in that group
            - **SMILES only**: Find structures everywhere
            - **Both**: Find structures in specific taxon

            **Chemical Search** (SACHEM/IDSM):
            - **Substructure**: Find compounds containing your structure
            - **Similarity**: Find similar compounds (Tanimoto 0.0-1.0, default 0.8)

            **Filters:** Mass (Da), Year, Formula (exact or element ranges + halogen control)

            **Export:** CSV, JSON, RDF/Turtle. Auto-compress >8MB.
            """)
        }
    )

    # Display side by side
    mo.hstack([url_api_section, help_section], gap=2, widths="equal")
    return


@app.cell
def url_params_check():
    # URL parameter detection and display
    _url_params_check = mo.query_params()

    # Display URL query info if parameters are present
    if _url_params_check and (
        "taxon" in _url_params_check or "smiles" in _url_params_check
    ):
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
        param_items = [
            f"- **{k}**: `{_url_params_check.get(k)}`"
            for k in known_params
            if k in _url_params_check
        ]
        if param_items:
            mo.callout(
                mo.md(f"""
                **URL Query Detected** - Auto-executing with: {chr(10).join(param_items)}
                """),
                kind="info",
            )
    return


@app.cell
def ui_search_params(search_params):
    ## TAXON INPUT
    taxon_input = mo.ui.text(
        value=search_params.taxon,
        label="🔬 Taxon Name or Wikidata QID - Optional",
        placeholder="e.g., Artemisia annua, Cinchona, Q157115, or * for all taxa",
        full_width=True,
    )

    ## SMILES INPUT
    smiles_input = mo.ui.text(
        value=search_params.smiles,
        label="🧪 Chemical Structure (SMILES) - Optional",
        placeholder="e.g., c1ccccc1 (benzene), CC(=O)Oc1ccccc1C(=O)O (aspirin)",
        full_width=True,
    )

    smiles_search_type = mo.ui.dropdown(
        options=["substructure", "similarity"],
        value=search_params.smiles_search_type,
        label="🔍 Chemical Search Type",
        full_width=True,
    )

    smiles_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=search_params.smiles_threshold,
        label="🎯 Similarity Threshold",
        full_width=True,
    )

    ## MASS FILTERS
    mass_filter = mo.ui.checkbox(
        label="⚖️ Filter by mass", value=search_params.mass_filter
    )

    mass_min = mo.ui.number(
        value=search_params.mass_min,
        start=0,
        stop=CONFIG["mass_ui_max"],
        step=0.001,
        label="Min mass (Da)",
        full_width=True,
    )
    mass_max = mo.ui.number(
        value=search_params.mass_max,
        start=0,
        stop=CONFIG["mass_ui_max"],
        step=0.001,
        label="Max mass (Da)",
        full_width=True,
    )

    formula_filter = mo.ui.checkbox(
        label="⚛️ Filter by molecular formula", value=search_params.formula_filter
    )
    exact_formula = mo.ui.text(
        value=search_params.exact_formula,
        label="Exact formula (e.g., C15H10O5)",
        placeholder="Leave empty to use element ranges",
        full_width=True,
    )

    # Element min/max inputs
    def _mk(e, mn, mx, k):
        return (
            mo.ui.number(
                value=mn,
                start=0,
                stop=CONFIG[k],
                label=f"{e} min",
                full_width=True,
            ),
            mo.ui.number(
                value=mx if mx is not None else CONFIG[k],
                start=0,
                stop=CONFIG[k],
                label=f"{e} max",
                full_width=True,
            ),
        )

    c_min, c_max = _mk("C", search_params.c_min, search_params.c_max, "element_c_max")
    h_min, h_max = _mk("H", search_params.h_min, search_params.h_max, "element_h_max")
    n_min, n_max = _mk("N", search_params.n_min, search_params.n_max, "element_n_max")
    o_min, o_max = _mk("O", search_params.o_min, search_params.o_max, "element_o_max")
    p_min, p_max = _mk("P", search_params.p_min, search_params.p_max, "element_p_max")
    s_min, s_max = _mk("S", search_params.s_min, search_params.s_max, "element_s_max")

    # Halogen selectors
    _ho = ["allowed", "required", "excluded"]
    f_state = mo.ui.dropdown(
        options=_ho, value=search_params.f_state, label="F", full_width=True
    )
    cl_state = mo.ui.dropdown(
        options=_ho, value=search_params.cl_state, label="Cl", full_width=True
    )
    br_state = mo.ui.dropdown(
        options=_ho, value=search_params.br_state, label="Br", full_width=True
    )
    i_state = mo.ui.dropdown(
        options=_ho, value=search_params.i_state, label="I", full_width=True
    )

    current_year = datetime.now().year
    year_filter = mo.ui.checkbox(
        label="🗓️ Filter by publication year", value=search_params.year_filter
    )
    year_start = mo.ui.number(
        value=search_params.year_start,
        start=CONFIG["year_range_start"],
        stop=current_year,
        label="Start year",
        full_width=True,
    )
    year_end = mo.ui.number(
        value=search_params.year_end,
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
    # Build structure search section (right column)
    structure_fields = [smiles_input, smiles_search_type]
    if smiles_search_type.value == "similarity":
        structure_fields.append(smiles_threshold)

    structure_section = mo.vstack(structure_fields)

    # Main search parameters: taxon on left, structure on right
    main_search = mo.hstack(
        [
            mo.vstack([taxon_input]),
            structure_section,
        ],
        gap=3,
        widths="equal",
    )

    # Optional filter buttons arranged horizontally
    filter_buttons = mo.hstack(
        [mass_filter, year_filter, formula_filter],
        gap=3,
        justify="start",
    )

    # Build filters UI
    filters_ui = [
        mo.md("## 🔍 Search Parameters"),
        main_search,
        mo.md("### Optional Filters"),
        filter_buttons,
    ]

    # Mass filter fields
    if mass_filter.value:
        filters_ui.append(mo.hstack([mass_min, mass_max], gap=2, widths="equal"))

    # Year filter fields
    if year_filter.value:
        filters_ui.append(mo.hstack([year_start, year_end], gap=2, widths="equal"))

    # Formula filter fields
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
    # Auto-run if URL parameters were detected, or if run button was clicked
    if not run_button.value and not search_params.auto_run:
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
                        search_mode="combined",
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
                    True,
                    mo.callout(mo.md(f"**Query Error:** {str(e)}"), kind="danger"),
                )
        elapsed = round(time.time() - start_time, 2)
        _ = mo.md(f"⏱️ Query completed in **{elapsed}s**.")
    return qid, results_df, taxon_warning


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
    results_df,
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
    if (not run_button.value and not search_params.auto_run) or results_df is None:
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
                        f"No natural products found for **{taxon_input.value}** ([{qid}]({SCHOLIA_URL}{qid})) with the current filters."
                    ),
                    kind="warn",
                )
            )
        summary_and_downloads = mo.vstack(parts) if len(parts) > 1 else parts[0]
    else:
        n_compounds = results_df.n_unique(subset=["compound"])
        n_taxa = results_df.n_unique(subset=["taxon"])
        n_refs = results_df.n_unique(subset=["reference"])
        n_entries = len(results_df)

        # Results header (on its own line)
        results_header = mo.md("## Results")

        # Taxon info
        if qid == "*":
            taxon_info = "All taxa"
        else:
            taxon_info = f"{taxon_input.value} [{qid}]({SCHOLIA_URL}{qid})"

        # Add SMILES search info if present
        if smiles_input.value and smiles_input.value.strip():
            _smiles_str = smiles_input.value.strip()
            search_type = smiles_search_type.value

            if search_type == "similarity":
                threshold_val = smiles_threshold.value
                smiles_info = f"SMILES: `{_smiles_str}` ({search_type}, threshold: {threshold_val})"
            else:
                smiles_info = f"SMILES: `{_smiles_str}...` ({search_type})"

            combined_info = f"{taxon_info} • {smiles_info}"
        else:
            combined_info = taxon_info

        # Search info
        search_info_display = mo.md(f"**{combined_info}**")

        # Provenance hash
        hash_info = mo.md(f"*Hashes:* Query: `{query_hash}` • Results: `{result_hash}`")

        # Stats cards - use list comprehension for DRY
        stats_data = [
            (n_compounds, "🧪", "Compound"),
            (n_taxa, "🌱", "Taxon"),
            (n_refs, "📚", "Reference"),
            (n_entries, "📝", "Entry"),
        ]
        stats_cards = mo.hstack(
            [
                mo.stat(
                    value=f"{n:,}",
                    label=f"{icon} {pluralize(name, n)}",
                    bordered=True,
                )
                for n, icon, name in stats_data
            ],
            gap=2,
            justify="start",
            wrap=False,
        )

        search_and_stats = mo.hstack(
            [
                mo.vstack([search_info_display, hash_info], gap=0.5),
                stats_cards,
            ],
            justify="space-between",
            align="center",
            gap=3,
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
    results_df,
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
    if (not run_button.value and not search_params.auto_run) or results_df is None:
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
        query_hash = None
        result_hash = None
    else:
        # Build filters for metadata (needed for hash computation)
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

        # Compute hashes for provenance using centralized helper
        query_hash, result_hash = compute_provenance_hashes(
            qid, taxon_input.value, active_filters, results_df
        )

        # Check if this is a large dataset BEFORE preparing export dataframes
        # This avoids creating copies of data in memory for large datasets
        taxon_name = taxon_input.value
        ui_is_large_dataset = len(results_df) > CONFIG["lazy_generation_threshold"]

        # For large datasets, defer export dataframe preparation to download time
        if ui_is_large_dataset:
            # Store reference to original df - export prep happens on-demand
            export_df = None
            export_df_rdf = None
        else:
            # For small datasets, prepare export dataframes immediately
            export_df = prepare_export_dataframe(results_df, include_rdf_ref=False)
            export_df_rdf = prepare_export_dataframe(results_df, include_rdf_ref=True)

        # Create metadata using already-built active_filters
        metadata = create_export_metadata(
            results_df if ui_is_large_dataset else export_df,
            taxon_input.value,
            qid,
            active_filters,
            query_hash,
            result_hash,
        )
        metadata_json = json.dumps(metadata, indent=2)
        citation_text = create_citation_text(taxon_input.value)
        # Display table data (apply row limit & depiction logic)
        total_rows = len(results_df)
        if total_rows > CONFIG["table_row_limit"]:
            limited_df = results_df.head(CONFIG["table_row_limit"])
            display_note = mo.callout(
                mo.md(
                    f"⚡ **Large Dataset Optimization**\n\n"
                    f"Your search returned **{total_rows:,} rows**. For optimal performance:\n"
                    f"- Displaying first **{CONFIG['table_row_limit']:,} rows** in table view\n"
                    f"- 2D structure images shown for first **{CONFIG['lazy_generation_threshold']:,}** rows only\n"
                    f"- Downloads are generated on-demand (click Generate buttons)\n"
                    f"- Export table disabled for large datasets"
                ),
                kind="info",
            )
        elif total_rows > CONFIG["lazy_generation_threshold"]:
            # Dataset fits in table but exceeds image/download limits
            limited_df = results_df
            display_note = mo.callout(
                mo.md(
                    f"**Optimized View**\n\n"
                    f"Your search returned **{total_rows:,} rows**. For smooth performance:\n"
                    f"- 2D structure images shown for first **{CONFIG['lazy_generation_threshold']:,}** rows (🧪 placeholder for others)\n"
                    f"- Downloads are generated on-demand (click Generate buttons)\n"
                    f"- SMILES strings available in all rows"
                ),
                kind="info",
            )
        else:
            display_note = mo.Html("")
            limited_df = results_df

        # Use different table component based on environment
        # mo.ui.table has issues in Pyodide/WASM, use plain HTML table instead
        # Build display DataFrame with HTML-formatted columns (shared for both WASM and native)
        display_df = build_display_dataframe(limited_df)

        if IS_PYODIDE:
            # In WASM: render as plain HTML table (GT as minimal formatting, TODO improve later)
            display_table = GT(display_df)

            # Export table - use simpler _repr_html_() for raw data view
            if not ui_is_large_dataset and export_df is not None:
                export_table_ui = GT(export_df)
            else:
                export_table_ui = mo.callout(
                    mo.md(
                        f"**Large Dataset ({len(results_df):,} rows)**\n\n"
                        f"Export table view is disabled for datasets over {CONFIG['lazy_generation_threshold']} rows.\n\n"
                        f"Use the download buttons to get your data."
                    ),
                    kind="info",
                )
        else:
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

            # Export table: only show for smaller datasets
            if not ui_is_large_dataset and export_df is not None:
                export_table_ui = mo.ui.table(
                    data=export_df,
                    selection=None,
                    page_size=CONFIG["page_size_export"],
                )
            else:
                export_table_ui = mo.callout(
                    mo.md(
                        f"**Large Dataset ({len(results_df):,} rows)**\n\n"
                        f"Export table view is disabled for datasets over {CONFIG['lazy_generation_threshold']} rows "
                        f"to ensure smooth performance.\n\n"
                        f"Use the download buttons above to get your data in CSV, JSON, or RDF format."
                    ),
                    kind="info",
                )

        # ALL downloads are lazy for large datasets - prevents iOS crashes
        buttons = []
        if ui_is_large_dataset:
            csv_generate_button = mo.ui.run_button(label="📄 Generate CSV")
            json_generate_button = mo.ui.run_button(label="📖 Generate JSON")
            rdf_generate_button = mo.ui.run_button(label="🐢 Generate RDF/Turtle")
            buttons.extend(
                [csv_generate_button, json_generate_button, rdf_generate_button]
            )
            # Store results_df - export dataframe will be prepared on-demand
            csv_generation_data = {
                "results_df": results_df,
                "active_filters": active_filters,
                "lazy": True,
            }
            json_generation_data = {
                "results_df": results_df,
                "active_filters": active_filters,
                "lazy": True,
            }
            rdf_generation_data = {
                "results_df": results_df,
                "taxon_input": taxon_input.value,
                "qid": qid,
                "active_filters": active_filters,
                "lazy": True,
            }
        else:
            csv_generate_button = None
            json_generate_button = None
            rdf_generate_button = None
            csv_generation_data = None
            json_generation_data = None
            rdf_generation_data = None
            # Only generate immediately for small datasets
            buttons.append(
                create_download_button(
                    export_df.write_csv(),
                    generate_filename(taxon_input.value, "csv", filters=active_filters),
                    "📥 CSV",
                    "text/csv",
                )
            )
            buttons.append(
                create_download_button(
                    export_df.write_json(),
                    generate_filename(
                        taxon_input.value, "json", filters=active_filters
                    ),
                    "📥 JSON",
                    "application/json",
                )
            )
            buttons.append(
                create_download_button(
                    export_to_rdf_turtle(
                        export_df_rdf, taxon_input.value, qid, active_filters
                    ),
                    generate_filename(taxon_input.value, "ttl", filters=active_filters),
                    "📥 RDF/Turtle",
                    "text/turtle",
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
                        "📥 Export View": export_table_ui,
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
        query_hash,
        rdf_generate_button,
        rdf_generation_data,
        result_hash,
        tables_ui,
        taxon_name,
        ui_is_large_dataset,
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
                generation_data["results_df"], include_rdf_ref=include_rdf_ref
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
            compressed_data, final_filename, final_mimetype = compress_if_large(
                raw_data,
                generate_filename(
                    taxon_name,
                    format_ext,
                    filters=generation_data["active_filters"],
                ),
            )

        return mo.vstack(
            [
                mo.callout(
                    mo.md(
                        f"✅ **{format_name} Ready** - {len(export_df):,} entries"
                        + (
                            " (compressed)"
                            if final_mimetype == "application/gzip"
                            else ""
                        )
                    ),
                    kind="success",
                ),
                mo.download(
                    data=compressed_data,
                    filename=final_filename,
                    label=f"📥 Download {format_name}",
                    mimetype=final_mimetype if final_mimetype else base_mimetype,
                ),
            ]
        )

    # CSV generation
    csv_download_ui = create_lazy_download_ui(
        csv_generate_button,
        csv_generation_data,
        "CSV",
        "csv",
        lambda df, d: df.write_csv(),
        "text/csv",
        ui_is_large_dataset,
    )

    # JSON generation
    json_download_ui = create_lazy_download_ui(
        json_generate_button,
        json_generation_data,
        "JSON",
        "json",
        lambda df, d: df.write_json(),
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
    _url_params = mo.query_params()
    search_params = SearchParams.from_url_params(_url_params)

    # Display auto-search message if URL parameters detected
    if search_params.auto_run:
        _ = mo.md(
            f"**Auto-executing search for:** {search_params.taxon if search_params.taxon else search_params.smiles}"
        )
    return (search_params,)


@app.cell
def footer():
    mo.md("""
    ---
    **Data:** <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> & <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a>  |
    **Code:** <a href="https://github.com/cdk/depict" style="color:#339966;">CDK Depict</a> & <a href="https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py" style="color:#339966;">lotus_wikidata_explorer.py</a>  |
    **License:** <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#006699;">CC0 1.0</a> for data & <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#006699;">AGPL-3.0</a> for code
    """)
    return


@app.function
def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # CLI mode - extract and reuse app.setup functions
        import argparse
        import gzip
        import io
        from pathlib import Path

        # Parse CLI arguments
        parser = argparse.ArgumentParser(description="Export LOTUS data")
        parser.add_argument("export")
        parser.add_argument("--taxon", help="Taxon name or QID")
        parser.add_argument("--output", "-o", help="Output file")
        parser.add_argument(
            "--format", "-f", choices=["csv", "json", "ttl"], default="csv"
        )
        parser.add_argument("--year-start", type=int, help="Minimum publication year")
        parser.add_argument("--year-end", type=int, help="Maximum publication year")
        parser.add_argument("--mass-min", type=float, help="Minimum molecular mass")
        parser.add_argument("--mass-max", type=float, help="Maximum molecular mass")

        # Molecular formula filters
        parser.add_argument(
            "--formula", help="Exact molecular formula (e.g., C15H10O5)"
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
            "--smiles", help="SMILES string for chemical structure search"
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
            "--compress", action="store_true", help="Compress output with gzip"
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
            "--verbose", "-v", action="store_true", help="Verbose output"
        )
        args = parser.parse_args()

        try:
            with open(__file__, "r", encoding="utf-8") as f:
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
            import re

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
                print(f"❌ Taxon not found: {args.taxon}", file=sys.stderr)
                sys.exit(1)

            # Determine search mode based on whether SMILES is provided
            search_mode = "taxon"
            if args.smiles:
                search_mode = "combined" if qid and qid != "*" else "smiles"

            # Improved verbose logging
            if args.verbose:
                print(f"🔍 Search Configuration:", file=sys.stderr)
                print(f"   Mode: {search_mode}", file=sys.stderr)
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
                    filters_applied.append(f"year ≥ {args.year_start}")
                if args.year_end:
                    filters_applied.append(f"year ≤ {args.year_end}")
                if args.mass_min:
                    filters_applied.append(f"mass ≥ {args.mass_min}")
                if args.mass_max:
                    filters_applied.append(f"mass ≤ {args.mass_max}")
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
                    print(f"   Filters: {', '.join(filters_applied)}", file=sys.stderr)
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
                formula_filt = create_formula_filters(
                    exact_formula=args.formula or "",
                    c_min=args.c_min or 0,
                    c_max=args.c_max or CONFIG["element_c_max"],
                    h_min=args.h_min or 0,
                    h_max=args.h_max or CONFIG["element_h_max"],
                    n_min=args.n_min or 0,
                    n_max=args.n_max or CONFIG["element_n_max"],
                    o_min=args.o_min or 0,
                    o_max=args.o_max or CONFIG["element_o_max"],
                    p_min=0,
                    p_max=CONFIG["element_p_max"],
                    s_min=0,
                    s_max=CONFIG["element_s_max"],
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
                search_mode=search_mode,
                smiles_search_type=args.smiles_search_type or "substructure",
                smiles_threshold=args.smiles_threshold,
            )

            if df.is_empty():
                print(f"❌ No data found", file=sys.stderr)
                sys.exit(1)

            if args.verbose:
                print(f"✅ Query Results:", file=sys.stderr)
                print(f"   Total entries: {len(df):,}", file=sys.stderr)

                # Show unique counts if columns exist
                if "compound" in df.columns:
                    unique_compounds = df.select(pl.col("compound")).n_unique()
                    print(f"   Unique compounds: {unique_compounds:,}", file=sys.stderr)
                if "taxon" in df.columns:
                    unique_taxa = df.select(pl.col("taxon")).n_unique()
                    print(f"   Unique taxa: {unique_taxa:,}", file=sys.stderr)
                if "reference" in df.columns:
                    unique_refs = df.select(pl.col("reference")).n_unique()
                    print(f"   Unique references: {unique_refs:,}", file=sys.stderr)
                print(file=sys.stderr)  # Empty line for readability

            # Build filters dict in the same format as the UI (needed for metadata and provenance)
            import json

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
                filters["molecular_formula"] = serialize_formula_filters(formula_filt)

            # Compute hashes for provenance (before showing metadata or exporting)
            # Query hash - based on search parameters (what was asked)
            query_components = [qid or "", args.taxon or ""]
            if filters:
                query_components.append(json.dumps(filters, sort_keys=True))
            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8")
            ).hexdigest()

            # Result hash - based on actual compound identifiers (what was found)
            compound_qids = sorted(
                [
                    extract_qid(row["compound"])
                    for row in df.iter_rows(named=True)
                    if row.get("compound")
                ]
            )
            result_hash = hashlib.sha256(
                "|".join(compound_qids).encode("utf-8")
            ).hexdigest()

            # Show metadata mode - use the REAL create_export_metadata function
            if args.show_metadata:
                # Use the REAL metadata function from app.setup with provenance hashes
                metadata = create_export_metadata(
                    df,
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
                    export_df, args.taxon, qid, filters if filters else None
                ).encode("utf-8")
            else:
                print(f"❌ Unknown format: {args.format}", file=sys.stderr)
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
                        f"✓ Exported {len(data):,} bytes to: {output_path}",
                        file=sys.stderr,
                    )

                # Export metadata if requested
                if args.export_metadata:
                    metadata = create_export_metadata(
                        df,
                        args.taxon,
                        qid,
                        filters if filters else None,
                        query_hash=query_hash,
                        result_hash=result_hash,
                    )
                    metadata_path = output_path.with_suffix(
                        output_path.suffix + ".metadata.json"
                    )
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    if args.verbose:
                        print(
                            f"✓ Exported metadata to: {metadata_path}",
                            file=sys.stderr,
                        )
            else:
                sys.stdout.buffer.write(data)

        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)
    else:
        app.run()


if __name__ == "__main__":
    main()
