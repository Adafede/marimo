"""
Data processing module for LOTUS Wikidata Explorer.

Handles data transformations, filtering, and DataFrame operations.
"""

from .processor import query_wikidata, prepare_export_dataframe
from .filters import (
    apply_mass_filter,
    apply_year_filter,
    apply_formula_filter,
    parse_molecular_formula,
    formula_matches_criteria,
)
from .transforms import (
    create_display_row,
    normalize_element_value,
    create_formula_filters,
)

__all__ = [
    # Processor
    "query_wikidata",
    "prepare_export_dataframe",
    # Filters
    "apply_mass_filter",
    "apply_year_filter",
    "apply_formula_filter",
    "parse_molecular_formula",
    "formula_matches_criteria",
    # Transforms
    "create_display_row",
    "normalize_element_value",
    "create_formula_filters",
]
