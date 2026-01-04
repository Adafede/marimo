"""
Data filtering functions for LOTUS Wikidata Explorer.

Handles mass, year, and molecular formula filtering of compound datasets.
"""

from functools import lru_cache
from typing import Optional
import polars as pl

from ..core.constants import SUBSCRIPT_MAP, FORMULA_PATTERN
from ..core.models import FormulaFilters

__all__ = [
    "apply_range_filter",
    "apply_year_filter",
    "apply_mass_filter",
    "parse_molecular_formula",
    "formula_matches_criteria",
    "apply_formula_filter",
]


def apply_range_filter(
    df: pl.DataFrame,
    column: str,
    min_val: Optional[float],
    max_val: Optional[float],
    extract_func=None,
) -> pl.DataFrame:
    """
    Generic range filter for DataFrame columns.

    Args:
        df: DataFrame to filter
        column: Column name to filter on
        min_val: Minimum value (inclusive), None for no minimum
        max_val: Maximum value (inclusive), None for no maximum
        extract_func: Optional function to extract value from column (e.g., .dt.year())

    Returns:
        Filtered DataFrame
    """
    if (min_val is None and max_val is None) or column not in df.columns:
        return df

    col_expr = pl.col(column)
    if extract_func:
        col_expr = extract_func(col_expr)

    # Build filter condition based on which bounds are set
    if min_val is not None and max_val is not None:
        # Both bounds set
        condition = (col_expr >= min_val) & (col_expr <= max_val)
    elif min_val is not None:
        # Only minimum bound set
        condition = col_expr >= min_val
    else:
        # Only maximum bound set
        condition = col_expr <= max_val

    return df.filter(pl.col(column).is_null() | condition)


def apply_year_filter(
    df: pl.DataFrame, year_start: Optional[int], year_end: Optional[int]
) -> pl.DataFrame:
    """Apply year range filter to publication dates."""
    return apply_range_filter(
        df, "pub_date", year_start, year_end, extract_func=lambda col: col.dt.year()
    )


def apply_mass_filter(
    df: pl.DataFrame, mass_min: Optional[float], mass_max: Optional[float]
) -> pl.DataFrame:
    """Apply mass range filter."""
    return apply_range_filter(df, "mass", mass_min, mass_max)


@lru_cache(maxsize=1024)
def parse_molecular_formula(formula: str) -> tuple:
    """
    Parse molecular formula and extract atom counts. Returns tuple for caching.

    Args:
        formula: Molecular formula string (may contain subscripts)

    Returns:
        Tuple of (element, count) pairs

    Example:
        >>> parse_molecular_formula("C₆H₁₂O₆")
        (('C', 6), ('H', 12), ('O', 6))
    """
    if not formula:
        return ()

    # Normalize formula by converting subscripts to regular numbers
    normalized_formula = formula.translate(SUBSCRIPT_MAP)

    # Pattern to match element followed by optional number
    matches = FORMULA_PATTERN.findall(normalized_formula)

    # Return tuple of (element, count) pairs for immutability and caching
    return tuple(
        (element, int(count) if count else 1) for element, count in matches if element
    )


def formula_matches_criteria(formula: str, filters: FormulaFilters) -> bool:
    """
    Check if a molecular formula matches the specified criteria.

    Args:
        formula: Molecular formula string
        filters: FormulaFilters object with filter criteria

    Returns:
        True if formula matches all criteria
    """
    # Early return: no formula means keep it (common case)
    if not formula:
        return True

    # Normalize formula once
    normalized_formula = formula.translate(SUBSCRIPT_MAP)

    # Early return: exact formula match (fast path)
    if filters.exact_formula and filters.exact_formula.strip():
        normalized_exact = filters.exact_formula.strip().translate(SUBSCRIPT_MAP)
        return normalized_formula == normalized_exact

    # Parse formula (cached for performance)
    atom_tuple = parse_molecular_formula(formula)
    atoms = dict(atom_tuple)

    # Check element ranges with early termination
    # Note: Using tuple for iteration efficiency
    elements_to_check = (
        ("C", filters.c),
        ("H", filters.h),
        ("N", filters.n),
        ("O", filters.o),
        ("P", filters.p),
        ("S", filters.s),
    )

    for element, elem_range in elements_to_check:
        if not elem_range.matches(atoms.get(element, 0)):
            return False  # Early termination

    # Check halogens with early termination
    halogens = (
        ("F", filters.f_state),
        ("Cl", filters.cl_state),
        ("Br", filters.br_state),
        ("I", filters.i_state),
    )

    for halogen, state in halogens:
        count = atoms.get(halogen, 0)
        if (state == "required" and count == 0) or (state == "excluded" and count > 0):
            return False  # Early termination

    return True


def apply_formula_filter(df: pl.DataFrame, filters: FormulaFilters) -> pl.DataFrame:
    """
    Apply molecular formula filters to the dataframe.

    Args:
        df: DataFrame with 'mf' (molecular formula) column
        filters: FormulaFilters object with filter criteria

    Returns:
        Filtered DataFrame
    """
    # Early return for efficiency
    if "mf" not in df.columns or not filters.is_active():
        return df

    # List comprehension is more efficient than building list with append
    mask = [
        formula_matches_criteria(row.get("mf", ""), filters)
        for row in df.iter_rows(named=True)
    ]

    return df.filter(pl.Series(mask))
