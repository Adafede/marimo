"""
DataFrame column transformations - requires polars.

Pure functions for transforming DataFrame columns.
"""

__all__ = [
    "rename_columns",
    "extract_from_url",
    "coalesce_columns",
    "parse_date_column",
    "cast_column",
]

from typing import Dict, Optional, List

import polars as pl


def rename_columns(
    df: pl.DataFrame,
    mapping: Dict[str, str],
) -> pl.DataFrame:
    """
    Rename DataFrame columns using a mapping.
    
    Only renames columns that exist in the DataFrame.
    
    Args:
        df: Input DataFrame
        mapping: Dict of {old_name: new_name}
    
    Returns:
        DataFrame with renamed columns
    
    Example:
        >>> df = pl.DataFrame({"a": [1], "b": [2]})
        >>> rename_columns(df, {"a": "x", "c": "y"})  # "c" doesn't exist, ignored
        shape: (1, 2)
        ┌─────┬─────┐
        │ x   │ b   │
        ├─────┼─────┤
        │ 1   │ 2   │
        └─────┴─────┘
    """
    existing = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(existing)


def extract_from_url(
    df: pl.DataFrame,
    column: str,
    prefix: str,
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Extract identifier from URL by removing prefix.
    
    Args:
        df: Input DataFrame
        column: Column containing URLs
        prefix: URL prefix to remove
        output_column: Name for new column (defaults to column + "_id")
    
    Returns:
        DataFrame with new column containing extracted ID
    
    Example:
        >>> df = pl.DataFrame({"url": ["http://example.com/entity/Q123"]})
        >>> extract_from_url(df, "url", "http://example.com/entity/", "qid")
        shape: (1, 2)
        ┌──────────────────────────────────┬──────┐
        │ url                              │ qid  │
        ├──────────────────────────────────┼──────┤
        │ http://example.com/entity/Q123   │ Q123 │
        └──────────────────────────────────┴──────┘
    """
    if column not in df.columns:
        return df
    
    out_col = output_column or f"{column}_id"
    return df.with_columns(
        pl.col(column).str.replace(prefix, "", literal=True).alias(out_col)
    )


def coalesce_columns(
    df: pl.DataFrame,
    columns: List[str],
    output_column: str,
    drop_source: bool = True,
) -> pl.DataFrame:
    """
    Combine multiple columns into one, using first non-null value.
    
    Args:
        df: Input DataFrame
        columns: List of column names to coalesce (in priority order)
        output_column: Name for output column
        drop_source: Whether to drop source columns
    
    Returns:
        DataFrame with coalesced column
    
    Example:
        >>> df = pl.DataFrame({"a": [None, "x"], "b": ["y", "z"]})
        >>> coalesce_columns(df, ["a", "b"], "result")
        shape: (2, 1)
        ┌────────┐
        │ result │
        ├────────┤
        │ y      │
        │ x      │
        └────────┘
    """
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return df
    
    if len(existing) == 1:
        df = df.with_columns(pl.col(existing[0]).alias(output_column))
    else:
        df = df.with_columns(pl.coalesce(existing).alias(output_column))
    
    if drop_source:
        df = df.drop([c for c in existing if c != output_column])
    
    return df


def parse_date_column(
    df: pl.DataFrame,
    column: str,
    format: str = "%Y-%m-%dT%H:%M:%SZ",
    output_type: str = "date",
) -> pl.DataFrame:
    """
    Parse string column to date/datetime.
    
    Args:
        df: Input DataFrame
        column: Column to parse
        format: strftime format string
        output_type: "date" or "datetime"
    
    Returns:
        DataFrame with parsed column
    """
    if column not in df.columns:
        return df
    
    parsed = pl.col(column).str.strptime(pl.Datetime, format=format, strict=False)
    
    if output_type == "date":
        parsed = parsed.dt.date()
    
    return df.with_columns(
        pl.when(pl.col(column).is_not_null() & (pl.col(column) != ""))
        .then(parsed)
        .otherwise(None)
        .alias(column)
    )


def cast_column(
    df: pl.DataFrame,
    column: str,
    dtype: pl.DataType,
    strict: bool = False,
) -> pl.DataFrame:
    """
    Cast column to specified dtype.
    
    Args:
        df: Input DataFrame
        column: Column to cast
        dtype: Target polars dtype
        strict: Whether to raise on cast errors
    
    Returns:
        DataFrame with casted column
    """
    if column not in df.columns:
        return df
    
    return df.with_columns(pl.col(column).cast(dtype, strict=strict))

