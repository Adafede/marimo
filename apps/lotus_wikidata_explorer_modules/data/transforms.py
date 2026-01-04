"""
Data transformation utilities.

Handles data format conversions, display row creation, and helper functions.
"""

from typing import Optional, Dict, Any
import polars as pl

try:
    import marimo as mo
except ImportError:
    mo = None

from ..core.config import CONFIG, ELEMENT_CONFIGS, HALOGEN_CONFIGS
from ..core.models import ElementRange, FormulaFilters
from ..core.utils import extract_qid

# Import UI helpers for creating links and images
# These are the canonical implementations used throughout the app
try:
    from ..ui.helpers import (
        create_link,
        create_wikidata_link,
        create_structure_image_url,
    )
except ImportError:
    # Fallback for non-UI contexts (won't be used in practice)
    create_link = None
    create_wikidata_link = None
    create_structure_image_url = None

__all__ = [
    "normalize_element_value",
    "create_formula_filters",
    "serialize_element_range",
    "serialize_formula_filters",
    "create_display_row",
]


def normalize_element_value(val: int, default: int) -> Optional[int]:
    """Normalize element value by converting default values to None."""
    return None if val == default else val


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


def serialize_element_range(element_range: ElementRange) -> Optional[Dict[str, int]]:
    """Convert ElementRange to dictionary for export, returns None if not active."""
    if not element_range.is_active():
        return None
    return {
        "min": element_range.min_val,
        "max": element_range.max_val,
    }


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


def create_display_row(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a display row for the table with images and links.

    Args:
        row: Dictionary with compound data

    Returns:
        Dictionary formatted for display with marimo components
    """
    if mo is None:
        raise ImportError("marimo is required for create_display_row")

    img_url = create_structure_image_url(row["smiles"])
    compound_qid = extract_qid(row["compound"])
    taxon_qid = extract_qid(row["taxon"])
    ref_qid = extract_qid(row["reference"])
    doi = row["ref_doi"]

    # Extract statement ID if available (for provenance transparency)
    statement_uri = row.get("statement", "")
    statement_id = statement_uri.split("/")[-1] if statement_uri else ""

    result = {
        "2D Depiction": mo.image(src=img_url),
        "Compound": row["name"],
        "Compound SMILES": row["smiles"],
        "Compound InChIKey": row["inchikey"],
        "Taxon": row["taxon_name"],
        "Reference title": row["ref_title"] or "-",
        "Reference DOI": create_link(f"https://doi.org/{doi}", doi)
        if doi
        else mo.Html("-"),
        "Compound QID": create_wikidata_link(
            compound_qid, color=CONFIG["color_wikidata_red"]
        ),
        "Taxon QID": create_wikidata_link(
            taxon_qid, color=CONFIG["color_wikidata_green"]
        ),
        "Reference QID": create_wikidata_link(
            ref_qid, color=CONFIG["color_wikidata_blue"]
        ),
        "Statement": create_link(statement_uri, statement_id)
        if statement_id
        else mo.Html("-"),
    }

    return result
