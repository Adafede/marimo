"""
UI module for LOTUS Wikidata Explorer.

Handles state management, taxon resolution, and UI helper functions.
"""

from .state import parse_url_state, should_auto_run, format_url_params_display
from .taxon_resolver import resolve_taxon_to_qid, create_taxon_warning_html
from .helpers import (
    create_link,
    create_wikidata_link,
    create_structure_image_url,
    pluralize,
    build_api_url,
    build_filters_dict,
    get_filter_values,
)

__all__ = [
    # State management
    "parse_url_state",
    "should_auto_run",
    "format_url_params_display",
    # Taxon resolution
    "resolve_taxon_to_qid",
    "create_taxon_warning_html",
    # Helpers
    "create_link",
    "create_wikidata_link",
    "create_structure_image_url",
    "pluralize",
    "build_api_url",
    "build_filters_dict",
    "get_filter_values",
]
