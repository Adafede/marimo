"""Filter DataFrame by value membership."""

__all__ = ["filter_by_values"]

import polars as pl


def filter_by_values(
    df: pl.DataFrame,
    column: str,
    values: list,
    exclude: bool = False,
) -> pl.DataFrame:
    """Filter DataFrame to rows where column matches (or excludes) given values."""
    if column not in df.columns:
        return df

    condition = pl.col(column).is_in(values)
    if exclude:
        condition = ~condition

    return df.filter(condition)
