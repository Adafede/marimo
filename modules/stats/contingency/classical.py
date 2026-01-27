"""
Classical statistics for 2×2 contingency tables.

Provides diagnostic metrics like sensitivity, specificity, precision, etc.
"""

import numpy as np


def sensitivity(
    a: np.ndarray | int,
    c: np.ndarray | int,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute sensitivity (recall, true positive rate).

    sensitivity = a / (a + c)

    Interpretation: Of all items in the group, what fraction have the feature?

    Args:
        a: True positives (feature present in group)
        c: False negatives (feature absent in group)
        continuity_correction: Added to denominator to avoid division by zero

    Returns:
        Sensitivity in [0, 1]

    Example:
        >>> sensitivity(8, 2)  # 8 with feature, 2 without, in group
        0.8  # 80% of group has the feature
    """
    numerator = np.asarray(a, dtype=np.float32)
    denominator = (
        np.asarray(a, dtype=np.float32)
        + np.asarray(c, dtype=np.float32)
        + continuity_correction
    )
    return np.clip(numerator / denominator, 0, 1)


def specificity(
    b: np.ndarray | int,
    d: np.ndarray | int,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute specificity (true negative rate).

    specificity = d / (b + d)

    Interpretation: Of all items outside the group, what fraction lack the feature?

    Args:
        b: False positives (feature present outside group)
        d: True negatives (feature absent outside group)
        continuity_correction: Added to denominator

    Returns:
        Specificity in [0, 1]

    Example:
        >>> specificity(5, 85)  # 5 with feature, 85 without, outside group
        0.944  # 94.4% of non-group items lack the feature
    """
    numerator = np.asarray(d, dtype=np.float32)
    denominator = (
        np.asarray(b, dtype=np.float32)
        + np.asarray(d, dtype=np.float32)
        + continuity_correction
    )
    return np.clip(numerator / denominator, 0, 1)


def precision(
    a: np.ndarray | int,
    b: np.ndarray | int,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute precision (positive predictive value).

    precision = a / (a + b)

    Interpretation: Of all items with the feature, what fraction are in the group?

    Args:
        a: True positives
        b: False positives
        continuity_correction: Added to denominator

    Returns:
        Precision in [0, 1]

    Example:
        >>> precision(8, 2)  # 8 in group, 2 outside group, both with feature
        0.8  # 80% of items with feature are in our group
    """
    numerator = np.asarray(a, dtype=np.float32)
    denominator = (
        np.asarray(a, dtype=np.float32)
        + np.asarray(b, dtype=np.float32)
        + continuity_correction
    )
    return np.clip(numerator / denominator, 0, 1)


def false_positive_rate(
    b: np.ndarray | int,
    d: np.ndarray | int,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute false positive rate (FPR = 1 - specificity).

    FPR = b / (b + d)

    Interpretation: Of all items outside the group, what fraction have the feature?

    Args:
        b: False positives
        d: True negatives
        continuity_correction: Added to denominator

    Returns:
        False positive rate in [0, 1]

    Example:
        >>> false_positive_rate(5, 85)
        0.056  # 5.6% of non-group items have the feature
    """
    numerator = np.asarray(b, dtype=np.float32)
    denominator = (
        np.asarray(b, dtype=np.float32)
        + np.asarray(d, dtype=np.float32)
        + continuity_correction
    )
    return np.clip(numerator / denominator, 0, 1)


def false_negative_rate(
    a: np.ndarray | int,
    c: np.ndarray | int,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute false negative rate (FNR = 1 - sensitivity).

    FNR = c / (a + c)

    Interpretation: Of all items in the group, what fraction lack the feature?

    Args:
        a: True positives
        c: False negatives
        continuity_correction: Added to denominator

    Returns:
        False negative rate in [0, 1]
    """
    numerator = np.asarray(c, dtype=np.float32)
    denominator = (
        np.asarray(a, dtype=np.float32)
        + np.asarray(c, dtype=np.float32)
        + continuity_correction
    )
    return np.clip(numerator / denominator, 0, 1)


def likelihood_ratio_positive(
    sens: np.ndarray | float = None,
    fpr: np.ndarray | float = None,
    a: np.ndarray | int = None,
    b: np.ndarray | int = None,
    c: np.ndarray | int = None,
    d: np.ndarray | int = None,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute positive likelihood ratio.

    LR+ = sensitivity / FPR

    Interpretation: How much more likely is the feature in the group vs outside?

    Can be computed from either:
    - Pre-computed sensitivity and FPR, OR
    - Contingency table cells a, b, c, d

    Args:
        sens: Pre-computed sensitivity (optional)
        fpr: Pre-computed false positive rate (optional)
        a, b, c, d: Contingency table cells (optional)
        continuity_correction: Added to denominators

    Returns:
        Positive likelihood ratio (≥0)

    Example:
        >>> # Sensitivity = 0.8, FPR = 0.05
        >>> likelihood_ratio_positive(sens=0.8, fpr=0.05)
        16.0  # 16× more likely in group than outside

        >>> # From contingency table
        >>> likelihood_ratio_positive(a=8, b=5, c=2, d=85)
        15.2  # Similar result
    """
    if sens is not None and fpr is not None:
        s = np.asarray(sens)
        f = np.asarray(fpr)
    elif all(x is not None for x in [a, b, c, d]):
        s = sensitivity(a, c, continuity_correction)
        f = false_positive_rate(b, d, continuity_correction)
    else:
        raise ValueError("Must provide either (sens, fpr) or (a, b, c, d)")

    return s / (f + continuity_correction)


def likelihood_ratio_negative(
    sens: np.ndarray | float = None,
    spec: np.ndarray | float = None,
    a: np.ndarray | int = None,
    b: np.ndarray | int = None,
    c: np.ndarray | int = None,
    d: np.ndarray | int = None,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute negative likelihood ratio.

    LR- = (1 - sensitivity) / specificity = FNR / specificity

    Interpretation: How much more likely is feature absence in group vs outside?

    Args:
        sens: Pre-computed sensitivity (optional)
        spec: Pre-computed specificity (optional)
        a, b, c, d: Contingency table cells (optional)
        continuity_correction: Added to denominators

    Returns:
        Negative likelihood ratio (≥0)

    Example:
        >>> likelihood_ratio_negative(sens=0.8, spec=0.95)
        0.21  # Feature absence is 0.21× as likely in group
    """
    if sens is not None and spec is not None:
        fnr = 1 - np.asarray(sens)
        sp = np.asarray(spec)
    elif all(x is not None for x in [a, b, c, d]):
        fnr = false_negative_rate(a, c, continuity_correction)
        sp = specificity(b, d, continuity_correction)
    else:
        raise ValueError("Must provide either (sens, spec) or (a, b, c, d)")

    return fnr / (sp + continuity_correction)


def f1_score(
    prec: np.ndarray | float = None,
    sens: np.ndarray | float = None,
    a: np.ndarray | int = None,
    b: np.ndarray | int = None,
    c: np.ndarray | int = None,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """
    Compute F1 score (harmonic mean of precision and sensitivity).

    F1 = 2 × (precision × sensitivity) / (precision + sensitivity)

    Args:
        prec: Pre-computed precision (optional)
        sens: Pre-computed sensitivity (optional)
        a, b, c: Contingency table cells (optional)
        continuity_correction: Added to denominators

    Returns:
        F1 score in [0, 1]

    Example:
        >>> f1_score(prec=0.8, sens=0.8)
        0.8
        >>> f1_score(prec=0.9, sens=0.5)
        0.643  # Penalizes imbalance
    """
    if prec is not None and sens is not None:
        p = np.asarray(prec)
        s = np.asarray(sens)
    elif all(x is not None for x in [a, b, c]):
        p = precision(a, b, continuity_correction)
        s = sensitivity(a, c, continuity_correction)
    else:
        raise ValueError("Must provide either (prec, sens) or (a, b, c)")

    return 2 * p * s / (p + s + continuity_correction)


def contingency_stats(
    a: np.ndarray | int,
    b: np.ndarray | int,
    c: np.ndarray | int,
    d: np.ndarray | int,
    continuity_correction: float = 0.5,
) -> dict[str, np.ndarray | float]:
    """
    Compute all classical statistics for a 2×2 contingency table.

    Args:
        a: True positives
        b: False positives
        c: False negatives
        d: True negatives
        continuity_correction: Added to denominators

    Returns:
        Dictionary with keys:
            - sensitivity, specificity, precision
            - fpr, fnr (false positive/negative rate)
            - lr_positive, lr_negative (likelihood ratios)
            - f1

    Example:
        >>> stats = contingency_stats(a=8, b=2, c=2, d=88)
        >>> stats['sensitivity']  # 0.8
        >>> stats['precision']    # 0.8
        >>> stats['lr_positive']  # ~35
    """
    sens = sensitivity(a, c, continuity_correction)
    spec = specificity(b, d, continuity_correction)
    prec = precision(a, b, continuity_correction)
    fpr = false_positive_rate(b, d, continuity_correction)
    fnr = false_negative_rate(a, c, continuity_correction)

    return {
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "fpr": fpr,
        "fnr": fnr,
        "lr_positive": sens / (fpr + continuity_correction),
        "lr_negative": fnr / (spec + continuity_correction),
        "f1": 2 * prec * sens / (prec + sens + continuity_correction),
    }
