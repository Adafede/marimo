"""Filter DataFrame by mass range."""

__all__ = ["filter_mass"]

from typing import Optional

import polars as pl

from .range import filter_range


def filter_mass(
    df: pl.DataFrame,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
    column: str = "mass",
) -> pl.DataFrame:
    """Filter DataFrame by mass range."""
    return filter_range(df, column, mass_min, mass_max)
