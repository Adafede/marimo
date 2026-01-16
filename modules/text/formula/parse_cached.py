"""Cached formula parsing."""

__all__ = ["parse_cached"]

from functools import lru_cache
from typing import Tuple

from .parse import parse


@lru_cache(maxsize=256)
def parse_cached(formula: str) -> Tuple[Tuple[str, int], ...]:
    """Parse molecular formula with caching. Returns tuple for hashability."""
    if not formula:
        return ()
    return tuple(parse(formula).items())
