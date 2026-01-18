"""Parse SPARQL response bytes into Polars DataFrame."""

__all__ = ["parse_sparql_response"]

import io
import json
import polars as pl


def parse_sparql_response(response_bytes: bytes) -> pl.DataFrame:
    """Parse SPARQL response bytes (CSV or JSON) into a Polars DataFrame.

    Optimized for streaming: checks only the first byte to detect format,
    then streams directly to Polars for CSV (most common case).

    Args:
        response_bytes: Raw bytes from SPARQL endpoint response

    Returns:
        Polars DataFrame with parsed results, or empty DataFrame if parsing fails
    """
    if not response_bytes:
        return pl.DataFrame()

    # Fast path: check first non-whitespace byte to detect format
    # This avoids stripping/copying the entire response
    first_byte = None
    for b in response_bytes:
        if b not in (32, 9, 10, 13):  # space, tab, newline, carriage return
            first_byte = b
            break

    if first_byte is None:
        return pl.DataFrame()

    # JSON response starts with '{' (0x7B)
    if first_byte == 0x7B:
        try:
            json_data = json.loads(response_bytes)
            bindings = json_data.get("results", {}).get("bindings", [])
            if not bindings:
                return pl.DataFrame()
            # Extract column names from first binding
            columns = list(bindings[0].keys())
            # Build columns directly instead of row-by-row
            data = {
                col: [b.get(col, {}).get("value", "") for b in bindings]
                for col in columns
            }
            return pl.DataFrame(data)
        except (json.JSONDecodeError, KeyError, IndexError):
            return pl.DataFrame()

    # CSV response - stream directly to Polars (most efficient path)
    try:
        return pl.read_csv(io.BytesIO(response_bytes))
    except Exception:
        return pl.DataFrame()
