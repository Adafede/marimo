"""
Export module for LOTUS Wikidata Explorer.

Handles data export in multiple formats (CSV, JSON, RDF/Turtle) with metadata.
"""

from .metadata import (
    create_export_metadata,
    create_dataset_hashes,
    create_citation_text,
)
from .formats import generate_filename, compress_if_large
from .rdf import export_to_rdf_turtle, build_dataset_description

__all__ = [
    # Metadata
    "create_export_metadata",
    "create_dataset_hashes",
    "create_citation_text",
    # Formats
    "generate_filename",
    "compress_if_large",
    # RDF
    "export_to_rdf_turtle",
    "build_dataset_description",
]
