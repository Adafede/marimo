"""Check if formula matches all filter criteria."""

__all__ = ["match_filters"]

from .filters import FormulaFilters
from .normalize import normalize
from .match_element import match_element
from .match_halogen import match_halogen


def match_filters(formula: str, filters: FormulaFilters) -> bool:
    """Check if a molecular formula matches all filter criteria."""
    if not formula:
        return True

    if filters.exact_formula and filters.exact_formula.strip():
        return normalize(formula) == normalize(filters.exact_formula.strip())

    for elem, range_filter in [("C", filters.c), ("H", filters.h), ("N", filters.n),
                                ("O", filters.o), ("P", filters.p), ("S", filters.s)]:
        if range_filter.is_active():
            if not match_element(formula, elem, range_filter.min_val, range_filter.max_val):
                return False

    for halogen, state in [("F", filters.f_state), ("Cl", filters.cl_state),
                           ("Br", filters.br_state), ("I", filters.i_state)]:
        if not match_halogen(formula, halogen, state):
            return False

    return True
