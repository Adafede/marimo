"""Base range filter for DataFrame."""

__all__ = ["filter_range"]

from collections.abc import Callable
from functools import reduce
from operator import and_
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

    col_expr = transform(pl.col(column)) if transform else pl.col(column)

    conditions = [
        cond
        for cond in (
            col_expr >= min_val if min_val is not None else None,
            col_expr <= max_val if max_val is not None else None,
        )
        if cond is not None
    ]

    if not conditions:
        return df

    combined = reduce(and_, conditions)

    if keep_nulls:
        combined = pl.col(column).is_null() | combined

    return df.filter(combined)
