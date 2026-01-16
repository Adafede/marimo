"""Check element range matching."""

__all__ = ["match_element"]

from typing import Optional

from .count_element import count_element


def match_element(formula: str, element: str, min_count: Optional[int] = None, max_count: Optional[int] = None) -> bool:
    """Check if element count in formula is within specified range."""
    cnt = count_element(formula, element)
    if min_count is not None and cnt < min_count:
        return False
    if max_count is not None and cnt > max_count:
        return False
    return True
