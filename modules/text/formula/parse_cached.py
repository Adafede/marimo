"""Cached formula parsing."""

__all__ = ["parse_cached"]

from functools import lru_cache

from .parse import parse

CACHE_SIZE: int = 256


@lru_cache(maxsize=CACHE_SIZE)
def parse_cached(formula: str) -> tuple[tuple[str, int], ...]:
    """Parse molecular formula with caching. Returns tuple for hashability."""
    if not formula:
        return ()
    return tuple(parse(formula).items())
