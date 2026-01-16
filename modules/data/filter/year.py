"""Filter DataFrame by year range on date column."""

__all__ = ["filter_year"]

import polars as pl

from .range import filter_range

DEFAULT_DATE_COLUMN: str = "pub_date"


def filter_year(
    df: pl.DataFrame,
    year_start: int | None = None,
    year_end: int | None = None,
    column: str = DEFAULT_DATE_COLUMN,
) -> pl.DataFrame:
    """Filter DataFrame by year range on a date column."""
    return filter_range(df, column, year_start, year_end, transform=lambda col: col.dt.year())
