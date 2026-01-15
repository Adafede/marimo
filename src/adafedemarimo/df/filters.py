"""
DataFrame filtering utilities - requires polars.

Pure functions for filtering DataFrames. No hidden configuration.
"""

__all__ = [
    "filter_range",
    "filter_by_values",
]

from typing import Optional, Callable, Any

# import polars as pl


def filter_range(
    df: pl.DataFrame,
    column: str,
    min_val: Optional[Any] = None,
    max_val: Optional[Any] = None,
    transform: Optional[Callable[[pl.Expr], pl.Expr]] = None,
    keep_nulls: bool = True,
) -> pl.DataFrame:
    """
    Filter DataFrame by column value range.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        min_val: Minimum value (inclusive), None for no minimum
        max_val: Maximum value (inclusive), None for no maximum
        transform: Optional transform to apply to column before comparison
                   (e.g., lambda col: col.dt.year() for date columns)
        keep_nulls: Whether to keep rows with null values in column
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> df = pl.DataFrame({"value": [1, 2, 3, 4, 5]})
        >>> filter_range(df, "value", min_val=2, max_val=4)
        shape: (3, 1)
        ┌───────┐
        │ value │
        ├───────┤
        │ 2     │
        │ 3     │
        │ 4     │
        └───────┘
    """
    if (min_val is None and max_val is None) or column not in df.columns:
        return df
    
    col_expr = pl.col(column)
    if transform:
        col_expr = transform(col_expr)
    
    # Build condition
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


def filter_by_values(
    df: pl.DataFrame,
    column: str,
    values: list,
    exclude: bool = False,
) -> pl.DataFrame:
    """
    Filter DataFrame to rows where column matches (or doesn't match) given values.
    
    Args:
        df: Input DataFrame
        column: Column name to filter
        values: List of values to match
        exclude: If True, exclude matching rows instead of keeping them
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> df = pl.DataFrame({"type": ["A", "B", "C", "A"]})
        >>> filter_by_values(df, "type", ["A", "B"])
        shape: (3, 1)
        ┌──────┐
        │ type │
        ├──────┤
        │ A    │
        │ B    │
        │ A    │
        └──────┘
    """
    if column not in df.columns:
        return df
    
    condition = pl.col(column).is_in(values)
    if exclude:
        condition = ~condition
    
    return df.filter(condition)

