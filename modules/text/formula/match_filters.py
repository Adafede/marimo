"""Check if formula matches all filter criteria."""

__all__ = ["match_filters"]

from .filters import FormulaFilters
from .match_element import match_element
from .match_halogen import match_halogen
from .normalize import normalize

# Element filter configuration: (element_symbol, filter_attribute)
_ELEMENT_FILTERS = (
    ("C", "c"),
    ("H", "h"),
    ("N", "n"),
    ("O", "o"),
    ("P", "p"),
    ("S", "s"),
)

# Halogen filter configuration: (halogen_symbol, state_attribute)
_HALOGEN_FILTERS = (
    ("F", "f_state"),
    ("Cl", "cl_state"),
    ("Br", "br_state"),
    ("I", "i_state"),
)


def check_element_filters(formula: str, filters: FormulaFilters) -> bool:
    """Check all element range filters."""
    return all(
        not getattr(filters, attr).is_active()
        or match_element(
            formula=formula,
            element=elem,
            min_count=getattr(filters, attr).min_val,
            max_count=getattr(filters, attr).max_val,
        )
        for elem, attr in _ELEMENT_FILTERS
    )


def check_halogen_filters(formula: str, filters: FormulaFilters) -> bool:
    """Check all halogen state filters."""
    return all(
        match_halogen(
            formula=formula,
            halogen=halogen,
            constraint=getattr(filters, attr),
        )
        for halogen, attr in _HALOGEN_FILTERS
    )


def match_filters(formula: str, filters: FormulaFilters) -> bool:
    """Check if a molecular formula matches all filter criteria."""
    if not formula:
        return True

    if filters.exact_formula and filters.exact_formula.strip():
        return normalize(formula) == normalize(filters.exact_formula.strip())

    return check_element_filters(formula, filters) and check_halogen_filters(
        formula,
        filters,
    )
