"""Resolve taxon matches from CSV data."""

__all__ = ["resolve_from_csv"]

import io

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def extract_qid_from_url(url: str) -> str:
    """Extract QID from Wikidata entity URL."""
    return url.split("/")[-1] if "/" in url else url


def parse_search_results(csv_bytes: bytes) -> list[tuple[str, str]]:
    """Parse search results CSV into list of (qid, name) tuples."""
    df = pl.scan_csv(source=io.BytesIO(csv_bytes))
    if df.is_empty():
        return []

    return [
        (extract_qid_from_url(row.get("taxon", "")), row.get("taxon_name", ""))
        for row in df.iter_rows(named=True)
        if row.get("taxon") and row.get("taxon_name")
    ]


def parse_connectivity(csv_bytes: bytes) -> dict[str, int]:
    """Parse connectivity CSV into qid -> compound_count mapping."""
    df = pl.scan_csv(source=io.BytesIO(csv_bytes))
    return {
        extract_qid_from_url(row.get("taxon", "")): int(
            row.get("compound_count", 0) or 0,
        )
        for row in df.iter_rows(named=True)
        if row.get("taxon")
    }


def parse_details(csv_bytes: bytes) -> dict[str, dict[str, str | None]]:
    """Parse details CSV into qid -> {description, parent} mapping."""
    df = pl.scan_csv(source=io.BytesIO(csv_bytes))
    return {
        extract_qid_from_url(row.get("taxon", "")): {
            "description": row.get("taxonDescription"),
            "parent": row.get("taxon_parentLabel"),
        }
        for row in df.iter_rows(named=True)
        if row.get("taxon")
    }


def resolve_from_csv(
    search_results_csv: bytes,
    connectivity_csv: bytes | None = None,
    details_csv: bytes | None = None,
) -> tuple[list[tuple[str, str, str | None, str | None, int | None]], dict[str, int]]:
    """
    Parse taxon search results and enrich with connectivity data.

    Returns:
        Tuple of:
        - List of matches: (qid, name, description, parent, compound_count)
        - Dict mapping qid -> compound_count
    """
    if not HAS_POLARS:
        raise ImportError("polars is required for taxon resolution")

    if not search_results_csv or not search_results_csv.strip():
        return [], {}

    base_matches = parse_search_results(csv_bytes=search_results_csv)
    if not base_matches:
        return [], {}

    connectivity_map = (
        parse_connectivity(csv_bytes=connectivity_csv)
        if connectivity_csv and connectivity_csv.strip()
        else {}
    )

    details_map = (
        parse_details(csv_bytes=details_csv)
        if details_csv and details_csv.strip()
        else {}
    )

    enriched = [
        (
            qid,
            name,
            details_map.get(qid, {}).get("description"),
            details_map.get(qid, {}).get("parent"),
            connectivity_map.get(qid),
        )
        for qid, name in base_matches
    ]

    return enriched, connectivity_map
