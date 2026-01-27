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
    """
    Compute credible interval for log fold-change relative to baseline.

    Transforms credible interval on probability scale [0, 1] to
    log fold-change scale: log_base(θ / baseline).

    Args:
        alpha: Posterior α
        beta: Posterior β
        baseline: Reference rate θ₀ for comparison
        probability: Coverage probability (default 0.89)
        log_base: Base for logarithm (default 2 for log₂)

    Returns:
        Tuple of (lower_fc, upper_fc) on log fold-change scale

    Example:
        >>> # Posterior: ~40% success rate, baseline: 10%
        >>> alpha, beta = 40, 60
        >>> fc_lower, fc_upper = fold_change_credible_interval(40, 60, 0.1)
        >>> # Returns approximately (1.5, 2.5) meaning 2^1.5 to 2^2.5 fold increase
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
    """
    Classify enrichment using ROPE (Region of Practical Equivalence).

    ROPE is the interval [-ε, +ε] on log fold-change scale where differences
    are considered practically negligible.

    Classification logic:
        - "enriched": CI entirely above +ε (strong evidence of enrichment)
        - "depleted": CI entirely below -ε (strong evidence of depletion)
        - "equivalent": CI entirely within [-ε, +ε] (negligible difference)
        - "undecided": CI overlaps ROPE boundaries (insufficient evidence)

    Args:
        alpha: Posterior α
        beta: Posterior β
        baseline: Reference rate θ₀
        rope_width: Half-width ε of ROPE on log scale (default 0.5 = 1.4x fold-change)
        ci_probability: CI coverage (default 0.89)
        log_base: Base for logarithm (default 2)

    Returns:
        Tuple of (decisions, p_above_rope, p_below_rope) where:
            - decisions: Classification string or array
            - p_above_rope: P(θ > baseline × base^ε)
            - p_below_rope: P(θ < baseline × base^(-ε))

    Example:
        >>> # Moderate enrichment: posterior ~30%, baseline 10%
        >>> alpha, beta = 30, 70
        >>> decisions, p_above, p_below = rope_decision(30, 70, 0.1, rope_width=0.5)
        >>> # Returns ("enriched", 0.99, 0.0) - strong evidence above ROPE
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
    """
    Compute comprehensive enrichment statistics.

    Returns posterior mean, mode, and fold-change metrics relative to baseline.

    Args:
        alpha: Posterior α
        beta: Posterior β
        baseline: Reference rate θ₀
        log_base: Base for logarithm (default 2)

    Returns:
        Dictionary with keys:
            - posterior_mean: E[θ]
            - posterior_mode: mode of θ
            - log_fold_change: log_base(E[θ] / baseline)
            - fold_change: E[θ] / baseline on original scale

    Example:
        >>> stats = enrichment_strength(30, 70, 0.1)
        >>> stats['posterior_mean']  # ~0.30
        >>> stats['log_fold_change']  # ~1.58 (log₂(3))
        >>> stats['fold_change']      # ~3.0
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
    """
    Blend parent posterior with global baseline for hierarchical prior.

    prior_center = w × parent_posterior + (1-w) × global_baseline

    where w controls how much we trust the parent's evidence.

    Args:
        parent_posterior_mean: Posterior mean from parent taxon/group
        global_baseline: Global dataset frequency θ₀
        hierarchical_weight: Weight for parent (0 = pure global, 1 = pure parent)
            Default 0.2 means 20% parent, 80% global

    Returns:
        Prior center for child's analysis

    Example:
        >>> # Parent taxon has 40% rate, global is 10%
        >>> hierarchical_prior_center(0.4, 0.1, hierarchical_weight=0.2)
        0.14  # 0.2 × 0.4 + 0.8 × 0.1
    """
    if not (0 <= hierarchical_weight <= 1):
        raise ValueError(
            f"hierarchical_weight must be in [0, 1], got {hierarchical_weight}",
        )

    w = hierarchical_weight
    parent = np.asarray(parent_posterior_mean)
    global_base = np.asarray(global_baseline)

    return w * parent + (1 - w) * global_base
