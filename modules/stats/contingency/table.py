"""Contingency table construction with diversity weighting.

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
    """Compute diversity-weighted count using geometric mean.

            weight = √(n_items × n_sources)

            This penalizes redundant observations (many items from one source)
            while rewarding diverse evidence (items from many sources).

    Parameters
    ----------
    n_items : np.ndarray | int
        N items.
    n_sources : np.ndarray | int
        N sources.

    Returns
    -------
    np.ndarray | int
        Rounded diversity-weighted counts as integer values.

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
    feature_total: np.ndarray | int | None,
    group_total: np.ndarray | int | None,
    universe_size: int | None,
    a_weight: np.ndarray | float | None = None,
) -> dict[str, np.ndarray | int]:
    """Construct 2×2 contingency table from marginals.

    Parameters
    ----------
    a_raw : np.ndarray | int
        A raw.
    feature_total : np.ndarray | int | None
        Feature total.
    group_total : np.ndarray | int | None
        Group total.
    universe_size : int | None
        Universe size.
    a_weight : np.ndarray | float | None
        None. Default is None.

    Returns
    -------
    dict[str, np.ndarray | int]
        Dictionary with contingency cells ``a``, ``b``, ``c``, ``c_eff``, ``d``, and total size ``N``.

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
    feature_total: np.ndarray | int | None = None,
    group_total: np.ndarray | int | None = None,
    universe_size: int | None = None,
    apply_diversity_weight: bool = True,
) -> dict[str, np.ndarray | int]:
    """Build contingency table with optional diversity weighting.

                Combines diversity_weight and contingency_table_2x2 in one call.

    Parameters
    ----------
    feature_in_group : np.ndarray | int
        Feature in group.
    n_sources : np.ndarray | int | None
        None. Default is None.
    feature_total : np.ndarray | int | None
        None. Default is None.
    group_total : np.ndarray | int | None
        None. Default is None.
    universe_size : int | None
        None. Default is None.
    apply_diversity_weight : bool
        True. Default is True.

    Returns
    -------
    dict[str, np.ndarray | int]
        Contingency table dictionary, including metadata such as ``a_raw`` and optionally ``n_sources``.

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
    """Compute observed rate (MLE) = a / (a + c).

                This is the maximum likelihood estimate of the rate,
                before any Bayesian shrinkage.

    Parameters
    ----------
    a : np.ndarray | int
        A.
    group_total : np.ndarray | int
        Group total.

    Returns
    -------
    np.ndarray | float
        Observed rates computed as ``a / group_total``.

    """
    numerator = np.asarray(a, dtype=np.float32)
    denominator = np.maximum(np.asarray(group_total, dtype=np.float32), 1e-10)
    return numerator / denominator


def baseline_rate(
    feature_total: np.ndarray | int,
    universe_size: int,
    min_rate: float = 1e-6,
) -> np.ndarray | float:
    """Compute baseline rate θ₀ = feature_total / N.

                This represents the expected rate if the group were a random
                sample from the universe.

    Parameters
    ----------
    feature_total : np.ndarray | int
        Feature total.
    universe_size : int
        Universe size.
    min_rate : float
        Default is 1e-06.

    Returns
    -------
    np.ndarray | float
        Baseline rates clipped to at least ``min_rate``.

    """
    rate = np.asarray(feature_total, dtype=np.float32) / universe_size
    return np.maximum(rate, min_rate)
