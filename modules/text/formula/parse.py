"""Parse molecular formula into element counts."""

__all__ = ["parse"]

import re

from .normalize import normalize

ELEMENT_PATTERN = re.compile(pattern=r"([A-Z][a-z]?)(\d*)")


def parse(formula: str) -> dict[str, int]:
    """Parse molecular formula into element counts.

    Parameters
    ----------
    formula : str
        Formula.

    Returns
    -------
    dict[str, int]
        Dictionary containing parse.
    """
    if not formula:
        return {}
    normalized = normalize(formula)
    matches = ELEMENT_PATTERN.findall(normalized)
    return {elem: int(count) if count else 1 for elem, count in matches if elem}
