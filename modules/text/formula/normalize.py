"""Normalize subscript digits to regular digits."""

__all__ = ["normalize"]

from .subscript_map import SUBSCRIPT_MAP


def normalize(formula: str) -> str:
    """Convert subscript digits to regular digits.

Parameters
----------
formula : str
    Formula.

Returns
-------
str
    Computed result.
    """
    return formula.translate(SUBSCRIPT_MAP)
