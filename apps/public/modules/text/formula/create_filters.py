"""Factory function to create FormulaFilters."""

__all__ = ["create_filters"]

from .element_range import ElementRange
from .filters import FormulaFilters


def create_filters(
    exact_formula: str = "",
    c_min: int = 0,
    c_max: int | None = None,
    h_min: int = 0,
    h_max: int | None = None,
    n_min: int = 0,
    n_max: int | None = None,
    o_min: int = 0,
    o_max: int | None = None,
    p_min: int = 0,
    p_max: int | None = None,
    s_min: int = 0,
    s_max: int | None = None,
    f_state: str = "allowed",
    cl_state: str = "allowed",
    br_state: str = "allowed",
    i_state: str = "allowed",
) -> FormulaFilters:
    """Factory function to create FormulaFilters from individual values."""
    return FormulaFilters(
        exact_formula=exact_formula.strip() if exact_formula else None,
        c=ElementRange(c_min, c_max),
        h=ElementRange(h_min, h_max),
        n=ElementRange(n_min, n_max),
        o=ElementRange(o_min, o_max),
        p=ElementRange(p_min, p_max),
        s=ElementRange(s_min, s_max),
        f_state=f_state,
        cl_state=cl_state,
        br_state=br_state,
        i_state=i_state,
    )
