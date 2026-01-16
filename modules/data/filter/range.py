"""Base range filter for DataFrame."""

__all__ = ["filter_range"]

from typing import Any, Callable, Optional

import polars as pl


def filter_range(
    df: pl.DataFrame,
    column: str,
    min_val: Optional[Any] = None,
    max_val: Optional[Any] = None,
    transform: Optional[Callable[[pl.Expr], pl.Expr]] = None,
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
