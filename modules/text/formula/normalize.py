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
        Return value produced by normalize.
    """
    return formula.translate(SUBSCRIPT_MAP)
