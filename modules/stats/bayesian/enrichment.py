"""
Bayesian enrichment analysis using fold-change and ROPE methodology.

ROPE = Region of Practical Equivalence: a region around the null value
where differences are considered negligible for practical purposes.
"""

import numpy as np
from .beta import (
    credible_interval,
    posterior_probability_above,
    posterior_probability_below,
)


def fold_change_credible_interval(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    baseline: np.ndarray | float,
    probability: float = 0.89,
    log_base: float = 2,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Compute credible interval for log fold-change relative to baseline.

    Transforms credible interval on probability scale [0, 1] to
    log fold-change scale: log_base(θ / baseline).

Parameters
----------
alpha : np.ndarray | float
    Alpha.
beta : np.ndarray | float
    Beta.
baseline : np.ndarray | float
    Baseline.
probability : float
    Default is 0.89.
log_base : float
    Default is 2.

Returns
-------
tuple[np.ndarray | float, np.ndarray | float]
    Computed result.
    """
    # Get CI on probability scale
    p_lower, p_upper = credible_interval(alpha, beta, probability)

    # Convert baseline and probabilities to arrays for vectorized operations
    t = np.maximum(np.asarray(baseline), 1e-10)
    p_l = np.maximum(np.asarray(p_lower), 1e-10)
    p_u = np.maximum(np.asarray(p_upper), 1e-10)

    # Transform to log fold-change scale
    fc_lower = np.log(p_l / t) / np.log(log_base)
    fc_upper = np.log(p_u / t) / np.log(log_base)

    return fc_lower, fc_upper


def rope_decision(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    baseline: np.ndarray | float,
    rope_width: float = 0.5,
    ci_probability: float = 0.89,
    log_base: float = 2,
) -> tuple[np.ndarray | str, np.ndarray | float, np.ndarray | float]:
    """Classify enrichment using ROPE (Region of Practical Equivalence).

    ROPE is the interval [-ε, +ε] on log fold-change scale where differences
    are considered practically negligible.

    Classification logic:
        - "enriched": CI entirely above +ε (strong evidence of enrichment)
        - "depleted": CI entirely below -ε (strong evidence of depletion)
        - "equivalent": CI entirely within [-ε, +ε] (negligible difference)
        - "undecided": CI overlaps ROPE boundaries (insufficient evidence)

Parameters
----------
alpha : np.ndarray | float
    Alpha.
beta : np.ndarray | float
    Beta.
baseline : np.ndarray | float
    Baseline.
rope_width : float
    Default is 0.5.
ci_probability : float
    Default is 0.89.
log_base : float
    Default is 2.

Returns
-------
tuple[np.ndarray | str, np.ndarray | float, np.ndarray | float]
    Computed result.
    """
    t = np.maximum(np.asarray(baseline), 1e-10)
    a = np.maximum(np.asarray(alpha), 1e-6)
    b = np.maximum(np.asarray(beta), 1e-6)

    # ROPE boundaries on probability scale
    # Upper: baseline × base^(+ε)
    # Lower: baseline × base^(-ε)
    upper_rope_prob = np.minimum(t * (log_base**rope_width), 1 - 1e-6)
    lower_rope_prob = np.maximum(t * (log_base ** (-rope_width)), 1e-6)

    # Posterior probabilities of being outside ROPE
    p_above = posterior_probability_above(a, b, upper_rope_prob)
    p_below = posterior_probability_below(a, b, lower_rope_prob)

    # Get CI on fold-change scale
    fc_lower, fc_upper = fold_change_credible_interval(
        a,
        b,
        t,
        ci_probability,
        log_base,
    )

    # Classify based on CI position relative to ROPE
    is_scalar = np.isscalar(fc_lower)

    if is_scalar:
        if fc_lower > rope_width:
            decision = "enriched"
        elif fc_upper < -rope_width:
            decision = "depleted"
        elif fc_lower > -rope_width and fc_upper < rope_width:
            decision = "equivalent"
        else:
            decision = "undecided"
    else:
        decision = np.where(
            fc_lower > rope_width,
            "enriched",
            np.where(
                fc_upper < -rope_width,
                "depleted",
                np.where(
                    (fc_lower > -rope_width) & (fc_upper < rope_width),
                    "equivalent",
                    "undecided",
                ),
            ),
        )

    return decision, p_above, p_below


def enrichment_strength(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    baseline: np.ndarray | float,
    log_base: float = 2,
) -> dict[str, np.ndarray | float]:
    """Compute comprehensive enrichment statistics.

    Returns posterior mean, mode, and fold-change metrics relative to baseline.

Parameters
----------
alpha : np.ndarray | float
    Alpha.
beta : np.ndarray | float
    Beta.
baseline : np.ndarray | float
    Baseline.
log_base : float
    Default is 2.

Returns
-------
dict[str, np.ndarray | float]
    Computed result.
    """
    from .beta import posterior_mean, posterior_mode

    a = np.asarray(alpha)
    b = np.asarray(beta)
    t = np.maximum(np.asarray(baseline), 1e-10)

    p_mean = posterior_mean(a, b)
    p_mode = posterior_mode(a, b)

    fc = p_mean / t
    log_fc = np.log(np.maximum(fc, 1e-10)) / np.log(log_base)

    return {
        "posterior_mean": p_mean,
        "posterior_mode": p_mode,
        "log_fold_change": log_fc,
        "fold_change": fc,
    }


def hierarchical_prior_center(
    parent_posterior_mean: np.ndarray | float,
    global_baseline: np.ndarray | float,
    hierarchical_weight: float = 0.2,
) -> np.ndarray | float:
    """Blend parent posterior with global baseline for hierarchical prior.

    prior_center = w × parent_posterior + (1-w) × global_baseline

    where w controls how much we trust the parent's evidence.

Parameters
----------
parent_posterior_mean : np.ndarray | float
    Parent posterior mean.
global_baseline : np.ndarray | float
    Global baseline.
hierarchical_weight : float
    Default is 0.2.

Returns
-------
np.ndarray | float
    Computed result.
    """
    if not (0 <= hierarchical_weight <= 1):
        raise ValueError(
            f"hierarchical_weight must be in [0, 1], got {hierarchical_weight}",
        )

    w = hierarchical_weight
    parent = np.asarray(parent_posterior_mean)
    global_base = np.asarray(global_baseline)

    return w * parent + (1 - w) * global_base
