"""Base range filter for DataFrame."""

__all__ = ["filter_range"]

from collections.abc import Callable
from typing import Any

import polars as pl


def filter_range(
    df: pl.DataFrame,
    column: str,
    min_val: Any | None = None,
    max_val: Any | None = None,
    transform: Callable[[pl.Expr], pl.Expr] | None = None,
    keep_nulls: bool = True,
) -> pl.DataFrame:
    """Filter DataFrame by column value range."""
    if (min_val is None and max_val is None) or column not in df.columns:
        return df

    col_expr = pl.col(column)
    if transform:
        col_expr = transform(col_expr)

    conditions = []
    if min_val is not None:
        conditions.append(col_expr >= min_val)
    if max_val is not None:
        conditions.append(col_expr <= max_val)

    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined & cond

    if keep_nulls:
        combined = pl.col(column).is_null() | combined

    return df.filter(combined)
