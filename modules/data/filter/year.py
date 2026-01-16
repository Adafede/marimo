"""Filter DataFrame by year range on date column."""

__all__ = ["filter_year"]

from typing import Optional

import polars as pl

from .range import filter_range


def filter_year(
    df: pl.DataFrame,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    column: str = "pub_date",
) -> pl.DataFrame:
    """Filter DataFrame by year range on a date column."""
    return filter_range(df, column, year_start, year_end, transform=lambda col: col.dt.year())
