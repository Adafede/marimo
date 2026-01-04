"""
UI helper functions for display and formatting.

Provides functions for creating links, formatting text, and building UI elements.
"""

from typing import Optional, Dict, Any
import marimo as mo
from urllib.parse import quote as url_quote, urlencode

from ..core.config import CONFIG, SCHOLIA_URL

__all__ = [
    "create_link",
    "create_wikidata_link",
    "create_structure_image_url",
    "pluralize",
    "build_api_url",
    "build_filters_dict",
    "get_filter_values",
]

# Pluralization map
PLURAL_MAP = {
    "Entry": "Entries",
    "entry": "entries",
    "Taxon": "Taxa",
    "taxon": "taxa",
    "Compound": "Compounds",
    "compound": "compounds",
}


def create_link(url: str, text: str, color="#3377c4") -> "mo.Html":
    """Create a styled hyperlink (requires marimo)."""
    if mo is None:
        raise ImportError("marimo is required for create_link")

    safe_text = text or url or ""
    safe_url = url or "#"
    return mo.Html(
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
        f'style="color: {color}; text-decoration: none; '
        f'border-bottom: 1px solid transparent; transition: border-color 0.2s;" '
        f"onmouseover=\"this.style.borderColor='{color}'\" "
        f"onmouseout=\"this.style.borderColor='transparent'\">{safe_text}</a>"
    )


def create_wikidata_link(qid: str, color: str = "#3377c4") -> mo.Html:
    """Create a Wikidata link for a QID."""
    return create_link(f"{SCHOLIA_URL}{qid}", qid, color=color) if qid else mo.Html("-")


def create_structure_image_url(smiles: Optional[str]) -> str:
    """
    Generate CDK Depict URL for molecular structure image.

    Args:
        smiles: SMILES string

    Returns:
        URL to structure image
    """
    if not smiles:
        return "https://via.placeholder.com/120x120?text=No+SMILES"
    encoded_smiles = url_quote(smiles)
    return f"{CONFIG['cdk_base']}?smi={encoded_smiles}&annotate=cip"


def get_filter_values(
    mass_filter,
    mass_min,
    mass_max,
    year_filter,
    year_start,
    year_end,
):
    """
    Extract conditional filter values based on whether filters are enabled.

    Args:
        mass_filter: Mass filter checkbox widget
        mass_min: Min mass widget
        mass_max: Max mass widget
        year_filter: Year filter checkbox widget
        year_start: Start year widget
        year_end: End year widget

    Returns:
        Tuple of (y_start, y_end, m_min, m_max)
    """
    y_start = year_start.value if year_filter.value else None
    y_end = year_end.value if year_filter.value else None
    m_min = mass_min.value if mass_filter.value else None
    m_max = mass_max.value if mass_filter.value else None

    return y_start, y_end, m_min, m_max


def build_filters_dict(
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    mass_filter,
    mass_min,
    mass_max,
    year_filter,
    year_start,
    year_end,
    formula_filter,
    exact_formula,
    c_min=None,
    c_max=None,
    h_min=None,
    h_max=None,
    n_min=None,
    n_max=None,
    o_min=None,
    o_max=None,
    p_min=None,
    p_max=None,
    s_min=None,
    s_max=None,
    f_state=None,
    cl_state=None,
    br_state=None,
    i_state=None,
) -> Dict[str, Any]:
    """
    Build filters dictionary from UI input values.

    This centralizes the filter-building logic that was duplicated across multiple cells.

    Args:
        smiles_input: SMILES input widget
        smiles_search_type: Search type widget
        smiles_threshold: Similarity threshold widget
        mass_filter: Mass filter checkbox
        mass_min: Min mass widget
        mass_max: Max mass widget
        year_filter: Year filter checkbox
        year_start: Start year widget
        year_end: End year widget
        formula_filter: Formula filter checkbox
        exact_formula: Exact formula widget
        c_min, c_max, etc: Element range widgets (optional)
        f_state, cl_state, etc: Halogen state widgets (optional)

    Returns:
        Dictionary of active filters
    """
    filters = {}

    # SMILES filter
    if smiles_input.value and smiles_input.value.strip():
        filters["chemical_structure"] = {
            "smiles": smiles_input.value.strip(),
            "search_type": smiles_search_type.value,
        }
        if smiles_search_type.value == "similarity":
            filters["chemical_structure"]["similarity_threshold"] = (
                smiles_threshold.value
            )

    # Mass filter
    if mass_filter.value:
        filters["mass"] = {"min": mass_min.value, "max": mass_max.value}

    # Year filter
    if year_filter.value:
        filters["year"] = {"start": year_start.value, "end": year_end.value}

    # Formula filter
    if formula_filter.value:
        formula_dict = {}

        if exact_formula.value:
            formula_dict["exact"] = exact_formula.value

        # Element ranges (if provided)
        if c_min is not None or c_max is not None:
            if c_min.value is not None or c_max.value is not None:
                formula_dict["c"] = {"min": c_min.value, "max": c_max.value}
        if h_min is not None or h_max is not None:
            if h_min.value is not None or h_max.value is not None:
                formula_dict["h"] = {"min": h_min.value, "max": h_max.value}
        if n_min is not None or n_max is not None:
            if n_min.value is not None or n_max.value is not None:
                formula_dict["n"] = {"min": n_min.value, "max": n_max.value}
        if o_min is not None or o_max is not None:
            if o_min.value is not None or o_max.value is not None:
                formula_dict["o"] = {"min": o_min.value, "max": o_max.value}
        if p_min is not None or p_max is not None:
            if p_min.value is not None or p_max.value is not None:
                formula_dict["p"] = {"min": p_min.value, "max": p_max.value}
        if s_min is not None or s_max is not None:
            if s_min.value is not None or s_max.value is not None:
                formula_dict["s"] = {"min": s_min.value, "max": s_max.value}

        # Halogen states (if provided)
        if any([f_state, cl_state, br_state, i_state]):
            formula_dict["halogens"] = {
                "f": f_state.value if f_state else "allowed",
                "cl": cl_state.value if cl_state else "allowed",
                "br": br_state.value if br_state else "allowed",
                "i": i_state.value if i_state else "allowed",
            }

        if formula_dict:
            filters["molecular_formula"] = formula_dict

    return filters


def pluralize(singular: str, count: int) -> str:
    """
    Return singular or plural form based on count.

    Args:
        singular: Singular form of word
        count: Count to determine pluralization

    Returns:
        Singular or plural form

    Example:
        >>> pluralize("Compound", 1)
        "Compound"
        >>> pluralize("Compound", 5)
        "Compounds"
        >>> pluralize("Taxon", 3)
        "Taxa"
    """
    return singular if count == 1 else PLURAL_MAP.get(singular, f"{singular}s")


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
    include_download: bool = False,
    download_format: str = "csv",
) -> str:
    """
    Build a shareable API URL from current search parameters.

    Args:
        taxon: Taxon name or QID
        smiles: SMILES structure
        smiles_search_type: "substructure" or "similarity"
        smiles_threshold: Similarity threshold (0.0-1.0)
        mass_filter: Whether mass filter is enabled
        mass_min: Minimum mass
        mass_max: Maximum mass
        year_filter: Whether year filter is enabled
        year_start: Start year
        year_end: End year
        formula_filter: Whether formula filter is enabled
        exact_formula: Exact molecular formula
        c_min: Minimum carbon count
        c_max: Maximum carbon count
        h_min: Minimum hydrogen count
        h_max: Maximum hydrogen count
        n_min: Minimum nitrogen count
        n_max: Maximum nitrogen count
        o_min: Minimum oxygen count
        o_max: Maximum oxygen count
        p_min: Minimum phosphorus count
        p_max: Maximum phosphorus count
        s_min: Minimum sulfur count
        s_max: Maximum sulfur count
        f_state: Fluorine state (allowed/required/excluded)
        cl_state: Chlorine state
        br_state: Bromine state
        i_state: Iodine state
        include_download: Whether to include download=true parameter
        download_format: Download format (csv, json, ttl)

    Returns:
        URL query string (e.g., "?taxon=Gentiana&mass_filter=true")
    """
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
        if c_min and c_min > 0:
            params["c_min"] = str(c_min)
        if c_max and c_max != CONFIG["element_c_max"]:
            params["c_max"] = str(c_max)
        if h_min and h_min > 0:
            params["h_min"] = str(h_min)
        if h_max and h_max != CONFIG["element_h_max"]:
            params["h_max"] = str(h_max)
        if n_min and n_min > 0:
            params["n_min"] = str(n_min)
        if n_max and n_max != CONFIG["element_n_max"]:
            params["n_max"] = str(n_max)
        if o_min and o_min > 0:
            params["o_min"] = str(o_min)
        if o_max and o_max != CONFIG["element_o_max"]:
            params["o_max"] = str(o_max)
        if p_min and p_min > 0:
            params["p_min"] = str(p_min)
        if p_max and p_max != CONFIG["element_p_max"]:
            params["p_max"] = str(p_max)
        if s_min and s_min > 0:
            params["s_min"] = str(s_min)
        if s_max and s_max != CONFIG["element_s_max"]:
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

    # Download parameters
    if include_download:
        params["download"] = "true"
        if download_format and download_format.lower() != "csv":
            params["format"] = download_format.lower()

    # Build URL
    if params:
        query_string = urlencode(params)
        return f"?{query_string}"
    else:
        return ""
