"""
Utility functions for LOTUS Wikidata Explorer.

Contains helper functions for QID extraction and data processing.
"""

from typing import Any, Dict
import polars as pl

from .config import CONFIG

__all__ = [
    "extract_qid",
    "extract_qids_from_dataframe",
    "get_binding_value",
]


def extract_qid(url: str) -> str:
    """Extract QID from a Wikidata entity URL."""
    if not url:
        return ""
    return url.rstrip("/").split("/")[-1]


def extract_qids_from_dataframe(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Extract QIDs from URLs in a DataFrame column."""
    if column not in df.columns:
        return df
    return df.with_columns(
        pl.col(column)
        .str.replace(r"^.*/([^/]+)$", r"$1")
        .alias(column.replace("_url", "_qid"))
    )


def get_binding_value(binding: Dict[str, Any], key: str, default: str = "") -> str:
    """Extract value from SPARQL binding dictionary."""
    return binding.get(key, {}).get("value", default)
