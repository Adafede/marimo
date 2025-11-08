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

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import polars as pl
    import re
    import time
    from datetime import datetime
    from functools import lru_cache
    from typing import Optional, Dict, Any
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

    # Shared SPARQL instance
    SPARQL = SPARQLWrapper("https://query.wikidata.org/sparql")
    SPARQL.setReturnFormat(JSON)
    SPARQL.addCustomHttpHeader("User-Agent", CONFIG["user_agent"])

    # Subscript translation map (constant for performance)
    SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    # Regex pattern for molecular formula parsing (compiled once)
    FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


@app.function
def build_taxon_search_query(taxon_name: str) -> str:
    """Build SPARQL query to find taxa by scientific name. Returns up to 10 results."""
    return f"""
    SELECT ?item ?name WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{taxon_name}";
                        mwapi:language "mul".
        ?item wikibase:apiOutputItem mwapi:item.
        ?num wikibase:apiOrdinal true.
      }}
      ?item wdt:P225 ?name.
    }}
    """


@app.function
def build_compounds_query(qid: str) -> str:
    return f"""
    SELECT DISTINCT ?structure ?structureLabel ?inchikey ?smiles_iso ?smiles_conn
                   ?taxon_name ?taxon ?ref_title ?ref_doi ?ref_qid ?pub_date ?mass ?mf
    WHERE {{
      ?taxon (wdt:P171*) wd:{qid};
             wdt:P225 ?taxon_name. 
      ?structure wdt:P235 ?inchikey;
                 wdt:P233 ?smiles_conn;
                 p:P703 ?statement.
      ?statement ps:P703 ?taxon;
                 prov:wasDerivedFrom ?ref.
      ?ref pr:P248 ?ref_qid.
      OPTIONAL {{ ?structure wdt:P2017 ?smiles_iso. }}
      OPTIONAL {{ ?structure wdt:P2067 ?mass. }}
      OPTIONAL {{ ?structure wdt:P274 ?mf. }}
      OPTIONAL {{
        SERVICE <https://query-scholarly.wikidata.org/sparql> {{
          ?ref_qid wdt:P1476 ?ref_title;
                   wdt:P356 ?ref_doi;
                   wdt:P577 ?pub_date.
        }}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {CONFIG["query_limit"]}
    """


@app.function
def execute_sparql(
    query: str, max_retries: int = CONFIG["max_retries"]
) -> Dict[str, Any]:
    """Execute SPARQL query with retry logic and exponential backoff."""
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
@lru_cache(maxsize=128)
def extract_qid(url: str) -> str:
    return url.replace("http://www.wikidata.org/entity/", "")


@app.function
@lru_cache(maxsize=1024)
def create_structure_image_url(smiles: str) -> str:
    if not smiles:
        return "https://via.placeholder.com/120x120?text=No+SMILES"
    encoded_smiles = quote(smiles)
    return f"{CONFIG['cdk_base']}?smi={encoded_smiles}&annotate=cip"


@app.function
@lru_cache(maxsize=256)
def resolve_taxon_to_qid(taxon_input: str) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve taxon name or QID to a valid QID.

    Returns:
        (qid, warning_message) where warning_message is None if no issues
    """
    taxon_input = taxon_input.strip()

    # Check if input is already a QID (Q followed by digits)
    if taxon_input.upper().startswith("Q") and taxon_input[1:].isdigit():
        return taxon_input.upper(), None

    # Otherwise, search for the taxon by name using EntitySearch
    try:
        query = build_taxon_search_query(taxon_input)
        results = execute_sparql(query)
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            return None, None

        # Extract all matching QIDs and their scientific names
        matches = []
        for binding in bindings:
            qid = extract_qid(binding.get("item", {}).get("value", ""))
            name = binding.get("name", {}).get("value", "")
            if qid and name:
                matches.append((qid, name))

        if not matches:
            return None, None

        # Check for exact name match (case-insensitive)
        exact_matches = [
            (qid, name) for qid, name in matches if name.lower() == taxon_input.lower()
        ]

        if len(exact_matches) == 1:
            return exact_matches[0][0], None
        elif len(exact_matches) > 1:
            # Multiple exact matches - ambiguous, return first with warning
            qid_list = ", ".join([qid for qid, _ in exact_matches])
            warning = f"Ambiguous taxon name. Multiple QIDs found: {qid_list}. Using {exact_matches[0][0]}. For precision, please use a specific QID."
            return exact_matches[0][0], warning

        # No exact match, use the first result
        if len(matches) > 1:
            # Show top matches for context
            qid_list = ", ".join([f"{qid} ({name})" for qid, name in matches[:3]])
            warning = f"No exact match. Similar taxa found: {qid_list}. Using {matches[0][0]}. For precision, use a QID directly."
            return matches[0][0], warning

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
    return (
        create_link(f"https://www.wikidata.org/wiki/{qid}", qid)
        if qid
        else mo.Html("—")
    )


@app.function
def pluralize(singular: str, count: int) -> str:
    """Return singular or plural form based on count with special cases."""
    if count == 1:
        return singular

    # Handle special plural forms
    special_plurals = {
        "Entry": "Entries",
        "entry": "entries",
        "Taxon": "Taxa",
        "taxon": "taxa",
    }

    return special_plurals.get(singular, f"{singular}s")


@app.function
def apply_year_filter(
    df: pl.DataFrame, year_start: Optional[int], year_end: Optional[int]
) -> pl.DataFrame:
    if year_start is None or year_end is None or "pub_date" not in df.columns:
        return df
    return df.filter(
        pl.col("pub_date").is_null()
        | (
            (pl.col("pub_date").dt.year() >= year_start)
            & (pl.col("pub_date").dt.year() <= year_end)
        )
    )


@app.function
def apply_mass_filter(
    df: pl.DataFrame, mass_min: Optional[float], mass_max: Optional[float]
) -> pl.DataFrame:
    if mass_min is None or mass_max is None or "mass" not in df.columns:
        return df
    return df.filter(
        pl.col("mass").is_null()
        | ((pl.col("mass") >= mass_min) & (pl.col("mass") <= mass_max))
    )


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
def formula_matches_criteria(
    formula: str,
    exact_formula: Optional[str],
    c_min: Optional[int],
    c_max: Optional[int],
    h_min: Optional[int],
    h_max: Optional[int],
    n_min: Optional[int],
    n_max: Optional[int],
    o_min: Optional[int],
    o_max: Optional[int],
    p_min: Optional[int],
    p_max: Optional[int],
    s_min: Optional[int],
    s_max: Optional[int],
    f_state: str,
    cl_state: str,
    br_state: str,
    i_state: str,
) -> bool:
    """Check if a molecular formula matches the specified criteria.

    Halogen states can be:
    - "allowed": no restriction (default)
    - "required": must be present (count > 0)
    - "excluded": must not be present (count = 0)
    """
    if not formula:
        return True  # Keep entries without formula

    # Normalize formula by converting subscripts to regular numbers
    normalized_formula = formula.translate(SUBSCRIPT_MAP)

    # If exact formula is specified, check for exact match
    if exact_formula and exact_formula.strip():
        normalized_exact = exact_formula.strip().translate(SUBSCRIPT_MAP)
        return normalized_formula == normalized_exact

    # Parse the formula (cached)
    atom_tuple = parse_molecular_formula(formula)
    atoms = dict(atom_tuple)

    # Check main elements ranges
    for element, min_val, max_val in [
        ("C", c_min, c_max),
        ("H", h_min, h_max),
        ("N", n_min, n_max),
        ("O", o_min, o_max),
        ("P", p_min, p_max),
        ("S", s_min, s_max),
    ]:
        if min_val is not None or max_val is not None:
            count = atoms.get(element, 0)
            if (min_val is not None and count < min_val) or (
                max_val is not None and count > max_val
            ):
                return False

    # Check halogens with state-based logic
    for halogen, state in [
        ("F", f_state),
        ("Cl", cl_state),
        ("Br", br_state),
        ("I", i_state),
    ]:
        count = atoms.get(halogen, 0)
        if (state == "required" and count == 0) or (state == "excluded" and count > 0):
            return False

    return True


@app.function
def apply_formula_filter(
    df: pl.DataFrame,
    exact_formula: Optional[str],
    c_min: Optional[int],
    c_max: Optional[int],
    h_min: Optional[int],
    h_max: Optional[int],
    n_min: Optional[int],
    n_max: Optional[int],
    o_min: Optional[int],
    o_max: Optional[int],
    p_min: Optional[int],
    p_max: Optional[int],
    s_min: Optional[int],
    s_max: Optional[int],
    f_state: str,
    cl_state: str,
    br_state: str,
    i_state: str,
) -> pl.DataFrame:
    """Apply molecular formula filters to the dataframe."""
    if "mf" not in df.columns:
        return df

    # If no filter is specified, return as is
    if (
        (exact_formula is None or not exact_formula.strip())
        and all(
            v is None
            for v in [
                c_min,
                c_max,
                h_min,
                h_max,
                n_min,
                n_max,
                o_min,
                o_max,
                p_min,
                p_max,
                s_min,
                s_max,
            ]
        )
        and all(state == "allowed" for state in [f_state, cl_state, br_state, i_state])
    ):
        return df

    # Apply filter with list comprehension (faster than loop + append)
    mask = [
        formula_matches_criteria(
            row.get("mf", ""),
            exact_formula,
            c_min,
            c_max,
            h_min,
            h_max,
            n_min,
            n_max,
            o_min,
            o_max,
            p_min,
            p_max,
            s_min,
            s_max,
            f_state,
            cl_state,
            br_state,
            i_state,
        )
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
    exact_formula: Optional[str] = None,
    c_min: Optional[int] = None,
    c_max: Optional[int] = None,
    h_min: Optional[int] = None,
    h_max: Optional[int] = None,
    n_min: Optional[int] = None,
    n_max: Optional[int] = None,
    o_min: Optional[int] = None,
    o_max: Optional[int] = None,
    p_min: Optional[int] = None,
    p_max: Optional[int] = None,
    s_min: Optional[int] = None,
    s_max: Optional[int] = None,
    f_state: str = "allowed",
    cl_state: str = "allowed",
    br_state: str = "allowed",
    i_state: str = "allowed",
) -> pl.DataFrame:
    query = build_compounds_query(qid)
    results = execute_sparql(query)
    bindings = results.get("results", {}).get("bindings", [])

    if not bindings:
        return pl.DataFrame()

    # Process results more efficiently
    rows = [
        {
            "structure": get_binding_value(b, "structure"),
            "name": get_binding_value(b, "structureLabel"),
            "inchikey": get_binding_value(b, "inchikey"),
            "smiles": get_binding_value(b, "smiles_iso")
            or get_binding_value(b, "smiles_conn"),
            "taxon_name": get_binding_value(b, "taxon_name"),
            "taxon": get_binding_value(b, "taxon"),
            "ref_title": get_binding_value(b, "ref_title"),
            "ref_doi": get_binding_value(b, "ref_doi").split("doi.org/")[-1]
            if (doi := get_binding_value(b, "ref_doi")) and doi.startswith("http")
            else get_binding_value(b, "ref_doi"),
            "reference": get_binding_value(b, "ref_qid"),
            "pub_date": get_binding_value(b, "pub_date", None),
            "mass": float(mass_raw)
            if (mass_raw := get_binding_value(b, "mass", None))
            else None,
            "mf": get_binding_value(b, "mf"),
        }
        for b in bindings
    ]

    df = pl.DataFrame(rows)

    # Optimize date conversion with single chain
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

    # Apply filters (only if they're active)
    df = apply_year_filter(df, year_start, year_end)
    df = apply_mass_filter(df, mass_min, mass_max)
    df = apply_formula_filter(
        df,
        exact_formula,
        c_min,
        c_max,
        h_min,
        h_max,
        n_min,
        n_max,
        o_min,
        o_max,
        p_min,
        p_max,
        s_min,
        s_max,
        f_state,
        cl_state,
        br_state,
        i_state,
    )

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
            .str.replace("http://www.wikidata.org/entity/", "", literal=True)
            .alias("compound_qid"),
            pl.col("taxon")
            .str.replace("http://www.wikidata.org/entity/", "", literal=True)
            .alias("taxon_qid"),
            pl.col("reference")
            .str.replace("http://www.wikidata.org/entity/", "", literal=True)
            .alias("reference_qid"),
        ]
    ).select(
        [
            pl.col("name").alias("compound_name"),
            pl.col("smiles").alias("compound_smiles"),
            pl.col("inchikey").alias("compound_inchikey"),
            "taxon_name",
            pl.col("ref_title").alias("reference_title"),
            pl.col("ref_doi").alias("reference_doi"),
            pl.col("pub_date").alias("reference_date"),
            "compound_qid",
            "taxon_qid",
            "reference_qid",
        ]
    )


@app.cell
def _():
    mo.md("""
    # 🌿 LOTUS Wikidata Explorer

    Explore chemical compounds from [LOTUS](https://doi.org/10.7554/eLife.70780) and 
    [Wikidata](https://www.wikidata.org/) for any taxon.

    Enter a taxon name to discover chemical compounds found in organisms of that taxonomic group.
    """)
    return


@app.cell
def _():
    ## MASS FILTERS
    mass_filter = mo.ui.checkbox(label="⚖ Filter by mass", value=False)

    mass_min = mo.ui.number(
        value=0, start=0, stop=10000, step=10, label="Min mass (Da)", full_width=True
    )

    mass_max = mo.ui.number(
        value=2000, start=0, stop=10000, step=10, label="Max mass (Da)", full_width=True
    )

    ## FORMULA FILTERS
    formula_filter = mo.ui.checkbox(label="⚛ Filter by molecular formula", value=False)

    exact_formula = mo.ui.text(
        value="",
        label="Exact formula (e.g., C15H10O5)",
        placeholder="Leave empty to use element ranges",
        full_width=True,
    )

    c_min = mo.ui.number(value=None, start=0, stop=100, label="C min", full_width=True)
    c_max = mo.ui.number(value=100, start=0, stop=100, label="C max", full_width=True)
    h_min = mo.ui.number(value=None, start=0, stop=200, label="H min", full_width=True)
    h_max = mo.ui.number(value=200, start=0, stop=200, label="H max", full_width=True)
    n_min = mo.ui.number(value=None, start=0, stop=50, label="N min", full_width=True)
    n_max = mo.ui.number(value=50, start=0, stop=50, label="N max", full_width=True)
    o_min = mo.ui.number(value=None, start=0, stop=50, label="O min", full_width=True)
    o_max = mo.ui.number(value=50, start=0, stop=50, label="O max", full_width=True)
    p_min = mo.ui.number(value=None, start=0, stop=20, label="P min", full_width=True)
    p_max = mo.ui.number(value=20, start=0, stop=20, label="P max", full_width=True)
    s_min = mo.ui.number(value=None, start=0, stop=20, label="S min", full_width=True)
    s_max = mo.ui.number(value=20, start=0, stop=20, label="S max", full_width=True)

    # Halogen selectors (allowed/required/excluded)
    halogen_options = ["allowed", "required", "excluded"]
    f_state = mo.ui.dropdown(
        options=halogen_options, value="allowed", label="F", full_width=True
    )
    cl_state = mo.ui.dropdown(
        options=halogen_options, value="allowed", label="Cl", full_width=True
    )
    br_state = mo.ui.dropdown(
        options=halogen_options, value="allowed", label="Br", full_width=True
    )
    i_state = mo.ui.dropdown(
        options=halogen_options, value="allowed", label="I", full_width=True
    )

    ## DATE FILTERS
    current_year = datetime.now().year

    taxon_input = mo.ui.text(
        value="Gentiana lutea",
        label="🔬 Taxon name or QID",
        placeholder="e.g., Swertia chirayita, Anabaena, Q157115, ...",
        full_width=True,
    )

    year_filter = mo.ui.checkbox(label="⏱ Filter by publication year", value=False)

    year_start = mo.ui.number(
        value=1900, start=1700, stop=current_year, label="Start year", full_width=True
    )

    year_end = mo.ui.number(
        value=current_year,
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
    taxon_input,
    year_end,
    year_filter,
    year_start,
):
    if not run_button.value:
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

                # Formula filters
                if formula_filter.value:
                    exact_f = (
                        exact_formula.value if exact_formula.value.strip() else None
                    )
                    _c_min = c_min.value
                    _c_max = c_max.value
                    _h_min = h_min.value
                    _h_max = h_max.value
                    _n_min = n_min.value
                    _n_max = n_max.value
                    _o_min = o_min.value
                    _o_max = o_max.value
                    _p_min = p_min.value
                    _p_max = p_max.value
                    _s_min = s_min.value
                    _s_max = s_max.value
                    _f_state = f_state.value
                    _cl_state = cl_state.value
                    _br_state = br_state.value
                    _i_state = i_state.value
                else:
                    exact_f = None
                    _c_min = _c_max = _h_min = _h_max = None
                    _n_min = _n_max = _o_min = _o_max = None
                    _p_min = _p_max = _s_min = _s_max = None
                    _f_state = _cl_state = _br_state = _i_state = "allowed"

                results_df = query_wikidata(
                    qid,
                    y_start,
                    y_end,
                    m_min,
                    m_max,
                    exact_f,
                    _c_min,
                    _c_max,
                    _h_min,
                    _h_max,
                    _n_min,
                    _n_max,
                    _o_min,
                    _o_max,
                    _p_min,
                    _p_max,
                    _s_min,
                    _s_max,
                    _f_state,
                    _cl_state,
                    _br_state,
                    _i_state,
                )
            except Exception as e:
                mo.stop(
                    True, mo.callout(mo.md(f"**Query Error:** {str(e)}"), kind="danger")
                )
        elapsed = round(time.time() - start_time, 2)
        mo.md(f"⏱️ Query completed in **{elapsed}s**.")
    return qid, results_df, taxon_warning


@app.cell
def _(qid, results_df, run_button, taxon_input, taxon_warning):
    if not run_button.value or results_df is None:
        summary_display = mo.Html("")
    elif len(results_df) == 0:
        # Show no compounds message, and taxon warning if present
        parts = []
        if taxon_warning:
            parts.append(mo.callout(mo.md(f"⚠️ {taxon_warning}"), kind="warn"))
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
            summary_parts.append(mo.callout(mo.md(f"⚠️ {taxon_warning}"), kind="warn"))

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
def _(results_df, run_button):
    if not run_button.value or results_df is None:
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
        export_table = mo.ui.table(
            export_df,
            selection=None,
            page_size=CONFIG["page_size_export"],
            show_column_summaries=False,
        )

        table_output = mo.vstack(
            [
                mo.md("### Data Tables"),
                mo.ui.tabs({"🖼️  Display": display_table, "📥 Export": export_table}),
            ]
        )

    table_output
    return


@app.cell
def _():
    mo.md(
        """
    ---
    **Data:** <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> & <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a>  |  
    **Code:** <a href="https://github.com/cdk/depict" style="color:#339966;">CDK Depict</a> & 
    <a href="https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py" style="color:#339966;">lotus_wikidata_explorer.py</a>  |  
    **License:** <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#006699;">CC0 1.0</a> for data & <a href="https://www.gnu.org/licenses/gpl-3.0.html" style="color:#006699;">GPL-3.0</a> for code
    """
    )
    return


if __name__ == "__main__":
    app.run()
