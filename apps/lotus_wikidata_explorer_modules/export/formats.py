"""
File format utilities for exports.

Handles filename generation, compression, and format-specific operations.
"""

from datetime import datetime
from typing import Dict, Any
import gzip

from ..core.config import CONFIG

__all__ = ["generate_filename", "compress_if_large"]


def generate_filename(
    taxon_name: str,
    file_type: str,
    prefix: str = "lotus_data",
    filters: Dict[str, Any] = None,
) -> str:
    """
    Generate standardized, descriptive filename for exports.

    Args:
        taxon_name: Name of taxon (or "*" for all)
        file_type: File extension (csv, json, ttl)
        prefix: Filename prefix
        filters: Active filters dictionary

    Returns:
        Standardized filename with date prefix

    Example:
        >>> generate_filename("Salix", "csv", filters={"mass": {...}})
        "20260104_lotus_data_Salix_filtered.csv"
    """
    # Handle wildcard for all taxa
    if taxon_name == "*":
        safe_name = "all_taxa"
    else:
        # Replace spaces and special characters
        safe_name = taxon_name.replace(" ", "_").replace("/", "_")

    # Build filename components
    components = [prefix, safe_name]

    # Add SMILES search type if present
    if filters and "chemical_structure" in filters:
        search_type = filters["chemical_structure"].get("search_type", "substructure")
        components.append(search_type)  # Just the type, not "smiles_type"

    # Add general filter indicator if other filters are active
    other_filters = {
        k: v for k, v in (filters or {}).items() if k != "chemical_structure"
    }
    if other_filters:
        components.append("filtered")

    date_str = datetime.now().strftime("%Y%m%d")
    filename_base = "_".join(components)
    return f"{date_str}_{filename_base}.{file_type}"


def compress_if_large(data: bytes, threshold: int = None) -> tuple[bytes, bool]:
    """
    Compress data if it exceeds threshold size.

    Args:
        data: Binary data to potentially compress
        threshold: Size threshold in bytes (uses CONFIG default if None)

    Returns:
        Tuple of (data, was_compressed)
    """
    if threshold is None:
        threshold = CONFIG["download_embed_threshold_bytes"]

    if len(data) > threshold:
        buffer = bytes()
        compressed = gzip.compress(data)
        return compressed, True

    return data, False
