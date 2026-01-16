"""Filter DataFrame by mass range."""

__all__ = ["filter_mass"]

import polars as pl

from .range import filter_range

DEFAULT_MASS_COLUMN: str = "mass"


def filter_mass(
    df: pl.DataFrame,
    mass_min: float | None = None,
    mass_max: float | None = None,
    column: str = DEFAULT_MASS_COLUMN,
) -> pl.DataFrame:
    """Filter DataFrame by mass range."""
    return filter_range(df, column, mass_min, mass_max)
