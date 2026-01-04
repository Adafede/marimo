"""
Main data processor for LOTUS Wikidata Explorer.

Orchestrates SPARQL queries, data transformation, and filtering.
"""

from typing import Optional
import polars as pl

from ..core.models import FormulaFilters
from ..core.constants import WIKIDATA_ENTITY_URL
from ..core.utils import get_binding_value
from ..query.builders import (
    build_smiles_taxon_query,
    build_smiles_similarity_query,
    build_smiles_substructure_query,
    build_all_compounds_query,
    build_compounds_query,
)
from ..query.executor import execute_sparql
from .filters import apply_year_filter, apply_mass_filter, apply_formula_filter

__all__ = ["query_wikidata", "prepare_export_dataframe"]


def query_wikidata(
    qid: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
    formula_filters: Optional[FormulaFilters] = None,
    smiles: Optional[str] = None,
    search_mode: str = "taxon",
    smiles_search_type: str = "substructure",
    smiles_threshold: float = 0.8,
) -> pl.DataFrame:
    """
    Query Wikidata for compounds associated to taxa using multiple search strategies.

    Supports three search modes:
    1. Taxon-only: Find all compounds in a taxonomic group
    2. SMILES-only: Find compounds by chemical structure (SACHEM)
    3. Combined: Find structures within a specific taxonomic group

    Args:
        qid: Wikidata QID of taxon (or "*" for all taxa)
        year_start: Minimum publication year
        year_end: Maximum publication year
        mass_min: Minimum molecular mass (Da)
        mass_max: Maximum molecular mass (Da)
        formula_filters: Molecular formula filtering criteria
        smiles: SMILES string for chemical structure search
        search_mode: "taxon", "smiles", or "combined"
        smiles_search_type: "substructure" or "similarity"
        smiles_threshold: Tanimoto similarity threshold (0.0-1.0)

    Returns:
        Polars DataFrame with compound data

    Raises:
        ValueError: For invalid search parameters
    """
    # Input validation
    if search_mode not in ("taxon", "smiles", "combined"):
        raise ValueError(
            f"Invalid search_mode: '{search_mode}'. "
            f"Must be one of: 'taxon', 'smiles', 'combined'"
        )

    if smiles_search_type not in ("substructure", "similarity"):
        raise ValueError(
            f"Invalid smiles_search_type: '{smiles_search_type}'. "
            f"Must be one of: 'substructure', 'similarity'"
        )

    if not (0.0 <= smiles_threshold <= 1.0):
        raise ValueError(
            f"Invalid smiles_threshold: {smiles_threshold}. "
            f"Must be between 0.0 and 1.0"
        )

    if year_start is not None and year_end is not None and year_start > year_end:
        raise ValueError(
            f"Invalid year range: start ({year_start}) > end ({year_end}). "
            f"Start year must be <= end year."
        )

    if mass_min is not None and mass_max is not None and mass_min > mass_max:
        raise ValueError(
            f"Invalid mass range: min ({mass_min}) > max ({mass_max}). "
            f"Minimum mass must be <= maximum mass."
        )

    # Build query based on search mode
    if search_mode == "combined" and smiles and qid:
        # Combined taxon + SMILES search
        query = build_smiles_taxon_query(
            smiles, qid, smiles_search_type, smiles_threshold
        )
    elif search_mode == "smiles" and smiles:
        # SMILES-only search
        if smiles_search_type == "similarity":
            query = build_smiles_similarity_query(smiles, smiles_threshold)
        else:  # Default to substructure
            query = build_smiles_substructure_query(smiles)
    elif qid == "*" or qid is None:
        query = build_all_compounds_query()
    else:
        query = build_compounds_query(qid)

    results = execute_sparql(query)
    bindings = results.get("results", {}).get("bindings", [])

    # Early return for empty results (efficiency - no DataFrame creation)
    if not bindings:
        return pl.DataFrame()

    # Process results efficiently with list comprehension (single pass)
    rows = [
        {
            "compound": get_binding_value(b, "compound"),
            "name": get_binding_value(b, "compoundLabel"),
            "inchikey": get_binding_value(b, "compound_inchikey"),
            "smiles": get_binding_value(b, "compound_smiles_iso")
            or get_binding_value(b, "compound_smiles_conn"),
            "taxon_name": get_binding_value(b, "taxon_name"),
            "taxon": get_binding_value(b, "taxon"),
            "ref_title": get_binding_value(b, "ref_title"),
            "ref_doi": (
                doi.split("doi.org/")[-1]
                if (doi := get_binding_value(b, "ref_doi")) and doi.startswith("http")
                else get_binding_value(b, "ref_doi")
            ),
            "reference": get_binding_value(b, "ref_qid"),
            "pub_date": get_binding_value(b, "ref_date", None),
            "mass": float(mass_raw)
            if (mass_raw := get_binding_value(b, "compound_mass", None))
            else None,
            "mf": get_binding_value(b, "compound_formula"),
            "statement": get_binding_value(b, "statement"),
            "ref": get_binding_value(b, "ref"),
        }
        for b in bindings
    ]

    # Create DataFrame once (efficiency - avoid copies)
    df = pl.DataFrame(rows)

    # Lazy transformations (Polars optimizes internally)
    if "pub_date" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("pub_date").is_not_null())
            .then(
                pl.col("pub_date")
                .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
                .dt.date()
            )
            .otherwise(None)
            .alias("pub_date")
        )

    # Chain filters for efficiency (Polars optimizes the execution plan)
    df = apply_year_filter(df, year_start, year_end)
    df = apply_mass_filter(df, mass_min, mass_max)

    if formula_filters:
        df = apply_formula_filter(df, formula_filters)

    # Final operations: deduplicate and sort
    # Note: unique() is efficient in Polars, keeps first occurrence
    return df.unique(subset=["compound", "taxon", "reference"], keep="first").sort(
        "name"
    )


def prepare_export_dataframe(
    df: pl.DataFrame,
    include_rdf_ref: bool = False,
) -> pl.DataFrame:
    """
    Prepare dataframe for export with cleaned QIDs and selected columns.

    Args:
        df: Input dataframe
        include_rdf_ref: If True, include ref URI for RDF export.
                        Statement is always included (for display and RDF).
                        Ref is only for RDF export (not needed in CSV/JSON/display).

    Returns:
        Cleaned DataFrame ready for export
    """
    # Extract QIDs for all entity columns
    df_with_qids = df.with_columns(
        [
            pl.col("compound")
            .str.replace(WIKIDATA_ENTITY_URL, "", literal=True)
            .alias("compound_qid"),
            pl.col("taxon")
            .str.replace(WIKIDATA_ENTITY_URL, "", literal=True)
            .alias("taxon_qid"),
            pl.col("reference")
            .str.replace(WIKIDATA_ENTITY_URL, "", literal=True)
            .alias("reference_qid"),
        ]
    )

    # Select and rename columns for export
    select_cols = [
        pl.col("name").alias("compound_name"),
        pl.col("smiles").alias("compound_smiles"),
        pl.col("inchikey").alias("compound_inchikey"),
        pl.col("mass").alias("compound_mass"),
        pl.col("mf").alias("molecular_formula"),
        "taxon_name",
        pl.col("ref_title").alias("reference_title"),
        pl.col("ref_doi").alias("reference_doi"),
        pl.col("pub_date").alias("reference_date"),
        "compound_qid",
        "taxon_qid",
        "reference_qid",
    ]

    # Always include statement (for display table and RDF)
    if "statement" in df_with_qids.columns:
        select_cols.append("statement")

    # Only include ref for RDF export (not needed for CSV/JSON/display)
    if include_rdf_ref and "ref" in df_with_qids.columns:
        select_cols.append("ref")

    return df_with_qids.select(select_cols)
