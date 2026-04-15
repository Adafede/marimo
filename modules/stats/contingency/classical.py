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
    """Compute sensitivity (recall, true positive rate).

            sensitivity = a / (a + c)

            Interpretation: Of all items in the group, what fraction have the feature?

    Parameters
    ----------
    a : np.ndarray | int
        A.
    c : np.ndarray | int
        C.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        Sensitivity values clipped to the interval ``[0, 1]``.
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
    """Compute specificity (true negative rate).

            specificity = d / (b + d)

            Interpretation: Of all items outside the group, what fraction lack the feature?

    Parameters
    ----------
    b : np.ndarray | int
        B.
    d : np.ndarray | int
        D.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        Specificity values clipped to the interval ``[0, 1]``.
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
    """Compute precision (positive predictive value).

            precision = a / (a + b)

            Interpretation: Of all items with the feature, what fraction are in the group?

    Parameters
    ----------
    a : np.ndarray | int
        A.
    b : np.ndarray | int
        B.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        Precision values clipped to the interval ``[0, 1]``.
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
    """Compute false positive rate (FPR = 1 - specificity).

            FPR = b / (b + d)

            Interpretation: Of all items outside the group, what fraction have the feature?

    Parameters
    ----------
    b : np.ndarray | int
        B.
    d : np.ndarray | int
        D.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        False-positive-rate values clipped to the interval ``[0, 1]``.
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
    """Compute false negative rate (FNR = 1 - sensitivity).

            FNR = c / (a + c)

            Interpretation: Of all items in the group, what fraction lack the feature?

    Parameters
    ----------
    a : np.ndarray | int
        A.
    c : np.ndarray | int
        C.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        False-negative-rate values clipped to the interval ``[0, 1]``.
    """
    numerator = np.asarray(c, dtype=np.float32)
    denominator = (
        np.asarray(a, dtype=np.float32)
        + np.asarray(c, dtype=np.float32)
        + continuity_correction
    )
    return np.clip(numerator / denominator, 0, 1)


def likelihood_ratio_positive(
    sens: np.ndarray | float | None = None,
    fpr: np.ndarray | float | None = None,
    a: np.ndarray | int | None = None,
    b: np.ndarray | int | None = None,
    c: np.ndarray | int | None = None,
    d: np.ndarray | int | None = None,
    continuity_correction: float = 0.5,
) -> np.ndarray | float:
    """Compute positive likelihood ratio.

            LR+ = sensitivity / FPR

            Interpretation: How much more likely is the feature in the group vs outside?

            Can be computed from either:
            - Pre-computed sensitivity and FPR, OR
            - Contingency table cells a, b, c, d

    Parameters
    ----------
    sens : np.ndarray | float | None
        None. Default is None.
    fpr : np.ndarray | float | None
        None. Default is None.
    a : np.ndarray | int | None
        None. Default is None.
    b : np.ndarray | int | None
        None. Default is None.
    c : np.ndarray | int | None
        None. Default is None.
    d : np.ndarray | int | None
        None. Default is None.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        Positive likelihood ratio (LR+) values.
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
    """Compute negative likelihood ratio.

            LR- = (1 - sensitivity) / specificity = FNR / specificity

            Interpretation: How much more likely is feature absence in group vs outside?

    Parameters
    ----------
    sens : np.ndarray | float
        None. Default is None.
    spec : np.ndarray | float
        None. Default is None.
    a : np.ndarray | int
        None. Default is None.
    b : np.ndarray | int
        None. Default is None.
    c : np.ndarray | int
        None. Default is None.
    d : np.ndarray | int
        None. Default is None.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        Negative likelihood ratio (LR-) values.
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
    """Compute F1 score (harmonic mean of precision and sensitivity).

            F1 = 2 × (precision × sensitivity) / (precision + sensitivity)

    Parameters
    ----------
    prec : np.ndarray | float
        None. Default is None.
    sens : np.ndarray | float
        None. Default is None.
    a : np.ndarray | int
        None. Default is None.
    b : np.ndarray | int
        None. Default is None.
    c : np.ndarray | int
        None. Default is None.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    np.ndarray | float
        F1 score values.
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
    """Compute all classical statistics for a 2×2 contingency table.

    Parameters
    ----------
    a : np.ndarray | int
        A.
    b : np.ndarray | int
        B.
    c : np.ndarray | int
        C.
    d : np.ndarray | int
        D.
    continuity_correction : float
        Default is 0.5.

    Returns
    -------
    dict[str, np.ndarray | float]
        LR+, LR-, and F1 metrics.
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
