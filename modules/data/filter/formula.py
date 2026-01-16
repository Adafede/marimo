"""Filter DataFrame by molecular formula criteria."""

__all__ = ["filter_formula"]

from collections.abc import Callable
from typing import Any

import polars as pl

DEFAULT_FORMULA_COLUMN: str = "mf"


def filter_formula(
    df: pl.DataFrame,
    filters: Any,
    column: str = DEFAULT_FORMULA_COLUMN,
    match_func: Callable[[str, Any], bool] | None = None,
) -> pl.DataFrame:
    """Filter DataFrame by molecular formula criteria."""
    if column not in df.columns:
        return df
    if hasattr(filters, "is_active") and not filters.is_active():
        return df
    if match_func is None:
        return df

    mask = [
        match_func(row.get(column, ""), filters) for row in df.iter_rows(named=True)
    ]
    return df.filter(pl.Series(mask))
