"""Normalize subscript digits to regular digits."""

__all__ = ["normalize"]

from .subscript_map import SUBSCRIPT_MAP


def normalize(formula: str) -> str:
    """Convert subscript digits to regular digits."""
    return formula.translate(SUBSCRIPT_MAP)
