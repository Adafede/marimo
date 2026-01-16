"""Factory function to create FormulaFilters."""

__all__ = ["create_filters"]

from typing import Optional

from .element_range import ElementRange
from .filters import FormulaFilters


def create_filters(
    exact_formula: str = "",
    c_min: int = 0, c_max: Optional[int] = None,
    h_min: int = 0, h_max: Optional[int] = None,
    n_min: int = 0, n_max: Optional[int] = None,
    o_min: int = 0, o_max: Optional[int] = None,
    p_min: int = 0, p_max: Optional[int] = None,
    s_min: int = 0, s_max: Optional[int] = None,
    f_state: str = "allowed",
    cl_state: str = "allowed",
    br_state: str = "allowed",
    i_state: str = "allowed",
) -> FormulaFilters:
    """Factory function to create FormulaFilters from individual values."""
    return FormulaFilters(
        exact_formula=exact_formula.strip() if exact_formula else None,
        c=ElementRange(c_min or None, c_max),
        h=ElementRange(h_min or None, h_max),
        n=ElementRange(n_min or None, n_max),
        o=ElementRange(o_min or None, o_max),
        p=ElementRange(p_min or None, p_max),
        s=ElementRange(s_min or None, s_max),
        f_state=f_state,
        cl_state=cl_state,
        br_state=br_state,
        i_state=i_state,
    )
