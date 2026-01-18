"""Parse SPARQL response bytes into Polars DataFrame."""

__all__ = ["parse_sparql_response"]

import io
import json
import polars as pl


def parse_sparql_response(response_bytes: bytes) -> pl.DataFrame:
    """Parse SPARQL response bytes (CSV or JSON) into a Polars DataFrame.

    Endpoints sometimes returns JSON despite requesting CSV format.
    This function detects the format and parses accordingly.

    Args:
        response_bytes: Raw bytes from SPARQL endpoint response

    Returns:
        Polars DataFrame with parsed results, or empty DataFrame if parsing fails
    """
    if not response_bytes or not response_bytes.strip():
        return pl.DataFrame()

    response_text = response_bytes.strip()

    # Detect JSON response (starts with '{')
    if response_text.startswith(b"{"):
        try:
            json_data = json.loads(response_bytes.decode("utf-8"))
            bindings = json_data.get("results", {}).get("bindings", [])
            if not bindings:
                return pl.DataFrame()
            # Convert SPARQL JSON bindings to flat dict rows
            rows = []
            for binding in bindings:
                row = {}
                for key, value in binding.items():
                    row[key] = value.get("value", "")
                rows.append(row)
            return pl.DataFrame(rows)
        except (json.JSONDecodeError, KeyError):
            return pl.DataFrame()
    else:
        # Parse as CSV
        try:
            return pl.read_csv(io.BytesIO(response_bytes))
        except Exception:
            return pl.DataFrame()
