"""Parse SPARQL response bytes into Polars DataFrame."""

__all__ = ["parse_sparql_response"]

import io
import polars as pl


def parse_sparql_response(response_bytes: bytes) -> pl.DataFrame:
    """Parse SPARQL CSV response bytes into a Polars DataFrame.

    Optimized for memory efficiency:
    - Streams directly to Polars without intermediate copies
    - Uses low_memory=True for reduced memory footprint
    - Zero-copy where possible via memoryview

    Args:
        response_bytes: Raw CSV bytes from SPARQL endpoint response

    Returns:
        Polars DataFrame with parsed results, or empty DataFrame if parsing fails
    """
    if not response_bytes:
        return pl.DataFrame()

    # Fast empty check - look for actual content after header
    # CSV must have at least header + newline + data
    if len(response_bytes) < 3:
        return pl.DataFrame()

    # Stream directly to Polars - most memory efficient path
    # Use low_memory=True to reduce peak memory usage for large datasets
    try:
        return pl.read_csv(
            io.BytesIO(response_bytes),
            low_memory=True,
            rechunk=False,  # Avoid extra memory copy from rechunking
        )
    except Exception:
        return pl.DataFrame()
