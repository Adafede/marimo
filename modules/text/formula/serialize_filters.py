"""Serialize FormulaFilters to dictionary."""

__all__ = ["serialize_filters"]

from typing import Any

from .filters import FormulaFilters
from .serialize_range import serialize_range

DEFAULT_ELEMENT_NAMES: dict[str, str] = {
    "c": "carbon",
    "h": "hydrogen",
    "n": "nitrogen",
    "o": "oxygen",
    "p": "phosphorus",
    "s": "sulfur",
}

DEFAULT_HALOGEN_NAMES: dict[str, str] = {
    "f": "fluorine",
    "cl": "chlorine",
    "br": "bromine",
    "i": "iodine",
}

# Element filter configuration: (key, attribute_name)
_ELEMENT_ATTRS = (
    ("c", "c"),
    ("h", "h"),
    ("n", "n"),
    ("o", "o"),
    ("p", "p"),
    ("s", "s"),
)

# Halogen filter configuration: (key, attribute_name)
_HALOGEN_ATTRS = (
    ("f", "f_state"),
    ("cl", "cl_state"),
    ("br", "br_state"),
    ("i", "i_state"),
)


def _serialize_elements(
    filters: FormulaFilters,
    element_names: dict[str, str],
) -> dict[str, dict[str, int | None]]:
    """Serialize element range filters to dictionary."""
    return {
        element_names.get(key, key): range_dict
        for key, attr in _ELEMENT_ATTRS
        if (range_dict := serialize_range(element_range=getattr(filters, attr)))
    }


def _serialize_halogens(
    filters: FormulaFilters,
    halogen_names: dict[str, str],
) -> dict[str, str]:
    """Serialize active halogen filters to dictionary."""
    return {
        halogen_names.get(key, key): getattr(filters, attr)
        for key, attr in _HALOGEN_ATTRS
        if getattr(filters, attr) != "allowed"
    }


def serialize_filters(
    filters: FormulaFilters | None,
    element_names: dict[str, str] | None = None,
    halogen_names: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Convert FormulaFilters to dictionary for metadata export."""
    if not filters or not filters.is_active():
        return None

    element_names = element_names or DEFAULT_ELEMENT_NAMES
    halogen_names = halogen_names or DEFAULT_HALOGEN_NAMES

    result: dict[str, Any] = {}

    if filters.exact_formula and filters.exact_formula.strip():
        result["exact_formula"] = filters.exact_formula.strip()

    result.update(_serialize_elements(filters=filters, element_names=element_names))

    active_halogens = _serialize_halogens(filters=filters, halogen_names=halogen_names)
    if active_halogens:
        result["halogens"] = active_halogens

    return result if result else None
