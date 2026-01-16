"""Filter DataFrame by molecular formula criteria."""

__all__ = ["filter_formula"]

from typing import Any, Callable, Optional

import polars as pl


def filter_formula(
    df: pl.DataFrame,
    filters: Any,
    column: str = "mf",
    match_func: Optional[Callable[[str, Any], bool]] = None,
) -> pl.DataFrame:
    """Filter DataFrame by molecular formula criteria."""
    if column not in df.columns:
        return df
    if hasattr(filters, 'is_active') and not filters.is_active():
        return df
    if match_func is None:
        return df

    mask = [match_func(row.get(column, ""), filters) for row in df.iter_rows(named=True)]
    return df.filter(pl.Series(mask))
