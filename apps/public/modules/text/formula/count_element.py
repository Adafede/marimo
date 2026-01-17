"""Count element occurrences in formula."""

__all__ = ["count_element"]

from .parse import parse


def count_element(formula: str, element: str) -> int:
    """Count occurrences of an element in a formula."""
    return parse(formula).get(element, 0)
