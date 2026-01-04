"""
UI state management from URL parameters.

Handles parsing URL query parameters and initializing application state.
"""

from typing import Optional, Dict, Any, List

__all__ = ["parse_url_state", "should_auto_run", "format_url_params_display"]


def _parse_bool(value: str) -> bool:
    """Parse boolean from URL parameter string."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_float(value: str, default: Optional[float] = None) -> Optional[float]:
    """Parse float from URL parameter string."""
    try:
        return float(value) if value else default
    except (ValueError, TypeError):
        return default


def _parse_int(value: str, default: Optional[int] = None) -> Optional[int]:
    """Parse integer from URL parameter string."""
    try:
        return int(value) if value else default
    except (ValueError, TypeError):
        return default


def parse_url_state(url_params: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse URL parameters into application state.

    Args:
        url_params: Dictionary of URL query parameters

    Returns:
        Dictionary of parsed state values
    """
    state = {
        # Main search inputs
        "taxon": url_params.get("taxon", ""),
        "smiles": url_params.get("smiles", ""),
        "smiles_search_type": url_params.get("smiles_search_type", "substructure"),
        "smiles_threshold": _parse_float(url_params.get("smiles_threshold", ""), 0.8),
        # Mass filter
        "mass_filter": _parse_bool(url_params.get("mass_filter", "false")),
        "mass_min": _parse_float(url_params.get("mass_min", ""), 0),
        "mass_max": _parse_float(url_params.get("mass_max", ""), 2000),
        # Year filter
        "year_filter": _parse_bool(url_params.get("year_filter", "false")),
        "year_start": _parse_int(url_params.get("year_start", ""), 1900),
        "year_end": _parse_int(url_params.get("year_end", ""), None),
        # Formula filter
        "formula_filter": _parse_bool(url_params.get("formula_filter", "false")),
        "exact_formula": url_params.get("exact_formula", ""),
        # Element ranges
        "c_min": _parse_int(url_params.get("c_min", ""), None),
        "c_max": _parse_int(url_params.get("c_max", ""), None),
        "h_min": _parse_int(url_params.get("h_min", ""), None),
        "h_max": _parse_int(url_params.get("h_max", ""), None),
        "n_min": _parse_int(url_params.get("n_min", ""), None),
        "n_max": _parse_int(url_params.get("n_max", ""), None),
        "o_min": _parse_int(url_params.get("o_min", ""), None),
        "o_max": _parse_int(url_params.get("o_max", ""), None),
        "p_min": _parse_int(url_params.get("p_min", ""), None),
        "p_max": _parse_int(url_params.get("p_max", ""), None),
        "s_min": _parse_int(url_params.get("s_min", ""), None),
        "s_max": _parse_int(url_params.get("s_max", ""), None),
        # Halogen states
        "f_state": url_params.get("f_state", "allowed"),
        "cl_state": url_params.get("cl_state", "allowed"),
        "br_state": url_params.get("br_state", "allowed"),
        "i_state": url_params.get("i_state", "allowed"),
        # Auto-download control
        "auto_download": _parse_bool(url_params.get("download", "false")),
        "download_format": url_params.get("format", "csv").lower(),
    }

    return state


def format_url_params_display(url_params: Dict[str, str]) -> Optional[str]:
    """
    Format URL parameters for display.

    Args:
        url_params: Dictionary of URL query parameters

    Returns:
        Formatted string for display, or None if no relevant params
    """
    if not url_params or not ("taxon" in url_params or "smiles" in url_params):
        return None

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
    param_items = []
    for key in known_params:
        if key in url_params:
            value = url_params.get(key)
            param_items.append(f"- **{key}**: `{value}`")

    if param_items:
        return "\n".join(param_items)
    else:
        return None


def should_auto_run(url_params: Dict[str, str]) -> bool:
    """
    Determine if search should auto-execute based on URL parameters.

    Args:
        url_params: Dictionary of URL query parameters

    Returns:
        True if search should auto-execute, False otherwise
    """
    # Auto-run if taxon or SMILES is provided
    return bool(url_params.get("taxon") or url_params.get("smiles"))
