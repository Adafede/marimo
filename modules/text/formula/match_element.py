"""Check element range matching."""

__all__ = ["match_element"]

from .count_element import count_element


def match_element(
    formula: str,
    element: str,
    min_count: int | None = None,
    max_count: int | None = None,
) -> bool:
    """Check if element count in formula is within specified range.

    Parameters
    ----------
    formula : str
        Formula.
    element : str
        Element.
    min_count : int | None
        None. Default is None.
    max_count : int | None
        None. Default is None.

    Returns
    -------
    bool
        Return value produced by match element.
    """
    cnt = count_element(formula, element)
    if min_count is not None and cnt < min_count:
        return False
    if max_count is not None and cnt > max_count:
        return False
    return True
