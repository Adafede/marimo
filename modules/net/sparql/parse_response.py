"""Parse SPARQL response bytes into Polars DataFrame."""

__all__ = ["parse_sparql_response"]

import io
import polars as pl


def parse_sparql_response(response_bytes: bytes) -> pl.LazyFrame:
    """Parse SPARQL CSV response bytes into a Polars LazyFrame.

    Memory-optimized implementation that returns a LazyFrame where possible:
    - Prefer `pl.scan_csv(tmp_path)` (LazyFrame) so callers can call `.collect()`
      when they need a materialized DataFrame. This minimizes peak memory.
    - If PyArrow is available, use PyArrow's CSV reader and convert to Polars,
      then return a LazyFrame via `.lazy()`.
    - Fall back to in-memory `pl.read_csv(...).lazy()` with low_memory and
      rechunk disabled.

    The function returns a LazyFrame (not materialized). Callers should use
    `.collect()` when they need an eager DataFrame.
    """
    if not response_bytes:
        return pl.DataFrame().lazy()

    # Fast empty check - look for actual content after header
    # CSV must have at least header + newline + data
    if len(response_bytes) < 3:
        return pl.DataFrame().lazy()

    # Stream directly to Polars - most memory efficient path
    # Use low_memory=True to reduce peak memory usage for large datasets
    try:
        # Return the LazyFrame directly; callers can `.collect()`.
        lf = pl.scan_csv(
            io.BytesIO(response_bytes),
            low_memory=True,
            rechunk=False,
        )
        return lf
    except Exception:
        pass

    try:
        return pl.read_csv(
            io.BytesIO(response_bytes),
            low_memory=True,
            rechunk=False,
            try_parse_dates=True,
            ignore_errors=True,
        ).lazy()
    except Exception:
        return pl.DataFrame().lazy()
