"""Serialize FormulaFilters to dictionary."""

__all__ = ["serialize_filters"]

from typing import Any

from .filters import FormulaFilters
from .serialize_range import serialize_range

DEFAULT_ELEMENT_NAMES: dict[str, str] = {
    "c": "carbon", "h": "hydrogen", "n": "nitrogen",
    "o": "oxygen", "p": "phosphorus", "s": "sulfur"
}
DEFAULT_HALOGEN_NAMES: dict[str, str] = {
    "f": "fluorine", "cl": "chlorine", "br": "bromine", "i": "iodine"
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

    for elem_key, elem_range in [("c", filters.c), ("h", filters.h), ("n", filters.n),
                                  ("o", filters.o), ("p", filters.p), ("s", filters.s)]:
        range_dict = serialize_range(elem_range)
        if range_dict:
            result[element_names.get(elem_key, elem_key)] = range_dict

    active_halogens = {}
    for hal_key, state in [("f", filters.f_state), ("cl", filters.cl_state),
                           ("br", filters.br_state), ("i", filters.i_state)]:
        if state != "allowed":
            active_halogens[halogen_names.get(hal_key, hal_key)] = state

    if active_halogens:
        result["halogens"] = active_halogens

    return result if result else None
