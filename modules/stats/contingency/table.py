"""
Contingency table construction with diversity weighting.

Handles 2×2 contingency tables for enrichment analysis:

              Feature Present  |  Feature Absent
    -----------------------------------------------
    In Group         a         |        b
    Not In Group     c         |        d

Where N = a + b + c + d is the total universe size.
"""

import numpy as np


def diversity_weight(
    n_items: np.ndarray | int,
    n_sources: np.ndarray | int,
) -> np.ndarray | int:
    """
    Compute diversity-weighted count using geometric mean.

    weight = √(n_items × n_sources)

    This penalizes redundant observations (many items from one source)
    while rewarding diverse evidence (items from many sources).

    Args:
        n_items: Number of items observed (e.g., compounds)
        n_sources: Number of independent sources (e.g., species)

    Returns:
        Diversity-weighted effective count (≥1)

    Example:
        >>> # 4 compounds from 1 species
        >>> diversity_weight(4, 1)
        2  # √(4 × 1) = 2

        >>> # 4 compounds from 4 species (more diverse)
        >>> diversity_weight(4, 4)
        4  # √(4 × 4) = 4

        >>> # 100 compounds from 1 species (redundant)
        >>> diversity_weight(100, 1)
        10  # √(100 × 1) = 10 (much less than 100)
    """
    items = np.asarray(n_items, dtype=np.float32)
    sources = np.asarray(n_sources, dtype=np.float32)

    weighted = np.sqrt(items * sources)
    result = np.round(weighted).astype(np.int32)

    # Ensure at least 1 if any items exist
    if np.isscalar(result):
        return max(1, result)
    return np.maximum(1, result)


def contingency_table_2x2(
    a_raw: np.ndarray | int,
    feature_total: np.ndarray | int,
    group_total: np.ndarray | int,
    universe_size: int,
    a_weight: np.ndarray | float | None = None,
) -> dict[str, np.ndarray | int]:
    """
    Construct 2×2 contingency table from marginals.

    Args:
        a_raw: Raw count in cell (feature present AND in group)
        feature_total: Total items with feature across universe
        group_total: Total items in group
        universe_size: Total universe size N
        a_weight: Optional weight for cell 'a' (default: use a_raw)
            If provided, also weights cell 'c' proportionally

    Returns:
        Dictionary with keys: a, b, c, d, N, c_eff
        Where c_eff is the weighted 'c' cell (if weighting applied)

    Example:
        >>> # Feature in 5 items, 3 of which are in our group
        >>> # Group has 20 items total, universe has 100 items
        >>> table = contingency_table_2x2(a_raw=3, feature_total=5,
        ...                                group_total=20, universe_size=100)
        >>> table['a'], table['b'], table['c'], table['d']
        (3, 2, 17, 78)
    """
    a_r = np.asarray(a_raw, dtype=np.int32)
    f_tot = np.asarray(feature_total, dtype=np.int32)
    g_tot = np.asarray(group_total, dtype=np.int32)

    # Use weighted 'a' if provided, otherwise raw count
    if a_weight is not None:
        a = np.asarray(a_weight, dtype=np.int32)
    else:
        a = a_r

    # Standard contingency table cells
    b = np.clip(f_tot - a_r, 0, None)
    c = np.clip(g_tot - a_r, 0, None)
    d = np.clip(universe_size - f_tot - g_tot + a_r, 0, None)

    # Weighted 'c' (proportional to weighting of 'a')
    if a_weight is not None:
        weight_ratio = a.astype(np.float32) / (a_r.astype(np.float32) + 1e-10)
        c_eff = np.round(c.astype(np.float32) * weight_ratio).astype(np.int32)
        c_eff = np.clip(c_eff, 0, None)
    else:
        c_eff = c

    return {
        "a": a,
        "b": b,
        "c": c,
        "c_eff": c_eff,
        "d": d,
        "N": universe_size,
    }


def contingency_from_presence(
    feature_in_group: np.ndarray | int,
    n_sources: np.ndarray | int | None = None,
    feature_total: np.ndarray | int = None,
    group_total: np.ndarray | int = None,
    universe_size: int = None,
    apply_diversity_weight: bool = True,
) -> dict[str, np.ndarray | int]:
    """
    Build contingency table with optional diversity weighting.

    Combines diversity_weight and contingency_table_2x2 in one call.

    Args:
        feature_in_group: Count of items with feature in group
        n_sources: Number of independent sources (for diversity weighting)
        feature_total: Total items with feature in universe
        group_total: Total items in group
        universe_size: Total universe size N
        apply_diversity_weight: If True and n_sources provided, apply weighting

    Returns:
        Dictionary with contingency table cells and metadata

    Example:
        >>> # 4 compounds with scaffold X from 4 different species in genus
        >>> # Scaffold X total: 10 compounds, Genus total: 30 compounds, Universe: 1000
        >>> table = contingency_from_presence(
        ...     feature_in_group=4, n_sources=4,
        ...     feature_total=10, group_total=30, universe_size=1000
        ... )
        >>> table['a']  # diversity-weighted: √(4 × 4) = 4
        4
    """
    a_raw = np.asarray(feature_in_group, dtype=np.int32)

    # Apply diversity weighting if requested and sources provided
    if apply_diversity_weight and n_sources is not None:
        a_weighted = diversity_weight(a_raw, n_sources)
    else:
        a_weighted = None

    table = contingency_table_2x2(
        a_raw=a_raw,
        feature_total=feature_total,
        group_total=group_total,
        universe_size=universe_size,
        a_weight=a_weighted,
    )

    # Include metadata
    table["a_raw"] = a_raw
    if n_sources is not None:
        table["n_sources"] = np.asarray(n_sources, dtype=np.int32)

    return table


def observed_rate(
    a: np.ndarray | int,
    group_total: np.ndarray | int,
) -> np.ndarray | float:
    """
    Compute observed rate (MLE) = a / (a + c).

    This is the maximum likelihood estimate of the rate,
    before any Bayesian shrinkage.

    Args:
        a: Count with feature in group
        group_total: Total in group (a + c)

    Returns:
        Observed proportion in [0, 1]

    Example:
        >>> observed_rate(3, 10)
        0.3
    """
    numerator = np.asarray(a, dtype=np.float32)
    denominator = np.maximum(np.asarray(group_total, dtype=np.float32), 1e-10)
    return numerator / denominator


def baseline_rate(
    feature_total: np.ndarray | int,
    universe_size: int,
    min_rate: float = 1e-6,
) -> np.ndarray | float:
    """
    Compute baseline rate θ₀ = feature_total / N.

    This represents the expected rate if the group were a random
    sample from the universe.

    Args:
        feature_total: Total items with feature
        universe_size: Total universe size N
        min_rate: Minimum rate to avoid division by zero

    Returns:
        Baseline proportion in [min_rate, 1]

    Example:
        >>> baseline_rate(50, 1000)
        0.05  # 5% of universe has this feature
    """
    rate = np.asarray(feature_total, dtype=np.float32) / universe_size
    return np.maximum(rate, min_rate)
