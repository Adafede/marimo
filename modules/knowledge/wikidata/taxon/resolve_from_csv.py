"""Resolve taxon matches from CSV data."""

__all__ = ["resolve_from_csv"]

import io

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


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

    df = pl.read_csv(io.BytesIO(search_results_csv))
    if df.is_empty():
        return [], {}

    matches = []
    for row in df.iter_rows(named=True):
        taxon_url = row.get("taxon", "")
        name = row.get("taxon_name", "")
        if taxon_url and name:
            qid = taxon_url.split("/")[-1] if "/" in taxon_url else taxon_url
            matches.append((qid, name, None, None, None))

    if not matches:
        return [], {}

    connectivity_map = {}
    if connectivity_csv and connectivity_csv.strip():
        conn_df = pl.read_csv(io.BytesIO(connectivity_csv))
        for row in conn_df.iter_rows(named=True):
            taxon_url = row.get("taxon", "")
            count = row.get("compound_count", 0)
            if taxon_url:
                qid = taxon_url.split("/")[-1] if "/" in taxon_url else taxon_url
                connectivity_map[qid] = int(count) if count else 0

    details_map = {}
    if details_csv and details_csv.strip():
        details_df = pl.read_csv(io.BytesIO(details_csv))
        for row in details_df.iter_rows(named=True):
            taxon_url = row.get("taxon", "")
            if taxon_url:
                qid = taxon_url.split("/")[-1] if "/" in taxon_url else taxon_url
                details_map[qid] = {
                    "description": row.get("taxonDescription"),
                    "parent": row.get("taxon_parentLabel"),
                }

    enriched = []
    for qid, name, _, _, _ in matches:
        details = details_map.get(qid, {})
        count = connectivity_map.get(qid)
        enriched.append((qid, name, details.get("description"), details.get("parent"), count))

    return enriched, connectivity_map
