"""
Beta distribution utilities for Bayesian inference.

Provides computations for Beta posterior distributions including:
- Posterior probabilities relative to thresholds
- Credible intervals (equal-tailed)
- Prior parameter computation
"""

import numpy as np
from scipy.special import betainc, betaincinv


def posterior_probability_above(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    threshold: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute P(θ > threshold | data) for Beta(α, β) posterior.

    Uses the regularized incomplete beta function:
        P(θ > t) = 1 - I_t(α, β)

    where I_t is the regularized incomplete beta function.

    Args:
        alpha: Shape parameter α (successes + prior)
        beta: Shape parameter β (failures + prior)
        threshold: Threshold value(s) on [0, 1]

    Returns:
        Posterior probability that θ exceeds threshold

    Example:
        >>> alpha, beta = 10, 5  # Observed 9 successes, 4 failures with weak prior
        >>> posterior_probability_above(alpha, beta, 0.5)
        0.9453125  # 94.5% chance θ > 0.5
    """
    a = np.maximum(np.asarray(alpha), 1e-6)
    b = np.maximum(np.asarray(beta), 1e-6)
    t = np.clip(np.asarray(threshold), 1e-6, 1 - 1e-6)
    return 1.0 - betainc(a, b, t)


def posterior_probability_below(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    threshold: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute P(θ < threshold | data) for Beta(α, β) posterior.

    Args:
        alpha: Shape parameter α (successes + prior)
        beta: Shape parameter β (failures + prior)
        threshold: Threshold value(s) on [0, 1]

    Returns:
        Posterior probability that θ is below threshold

    Example:
        >>> alpha, beta = 2, 10  # Observed 1 success, 9 failures with weak prior
        >>> posterior_probability_below(alpha, beta, 0.5)
        0.9990234375  # 99.9% chance θ < 0.5
    """
    a = np.maximum(np.asarray(alpha), 1e-6)
    b = np.maximum(np.asarray(beta), 1e-6)
    t = np.clip(np.asarray(threshold), 1e-6, 1 - 1e-6)
    return betainc(a, b, t)


def credible_interval(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    probability: float = 0.89,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Compute equal-tailed credible interval for Beta(α, β).

    Returns the interval [L, U] such that:
        P(θ < L) = (1 - probability) / 2
        P(θ > U) = (1 - probability) / 2

    For symmetric distributions, this equals the HDI (highest density interval).

    Args:
        alpha: Shape parameter α
        beta: Shape parameter β
        probability: Coverage probability (default 0.89 for 89% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) on [0, 1]

    Example:
        >>> alpha, beta = 30, 70  # Posterior from 29 successes, 69 failures
        >>> lower, upper = credible_interval(alpha, beta, 0.89)
        >>> # Returns (0.21, 0.39) - 89% sure θ is in this range
    """
    a = np.maximum(np.asarray(alpha), 0.01)
    b = np.maximum(np.asarray(beta), 0.01)
    tail = (1 - probability) / 2
    lower = betaincinv(a, b, tail)
    upper = betaincinv(a, b, 1 - tail)
    return lower, upper


def posterior_mean(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute posterior mean E[θ] = α / (α + β).

    Args:
        alpha: Shape parameter α
        beta: Shape parameter β

    Returns:
        Posterior mean on [0, 1]
    """
    a = np.asarray(alpha)
    b = np.asarray(beta)
    return a / (a + b)


def posterior_mode(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute posterior mode (MAP estimate).

    Mode = (α - 1) / (α + β - 2) when α > 1 and β > 1.
    Otherwise returns the mean as a fallback.

    Args:
        alpha: Shape parameter α
        beta: Shape parameter β

    Returns:
        Posterior mode on [0, 1]
    """
    a = np.asarray(alpha)
    b = np.asarray(beta)

    # Mode only defined when both α > 1 and β > 1
    valid_mode = (a > 1) & (b > 1)

    if np.isscalar(a):
        if valid_mode:
            return (a - 1) / (a + b - 2)
        return a / (a + b)

    result = np.where(valid_mode, (a - 1) / (a + b - 2), a / (a + b))
    return np.clip(result, 0, 1)


def posterior_variance(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute posterior variance Var[θ] = αβ / [(α+β)²(α+β+1)].

    Args:
        alpha: Shape parameter α
        beta: Shape parameter β

    Returns:
        Posterior variance
    """
    a = np.asarray(alpha)
    b = np.asarray(beta)
    ab_sum = a + b
    return (a * b) / (ab_sum**2 * (ab_sum + 1))


def beta_prior_params(
    mean: float,
    strength: float,
) -> tuple[float, float]:
    """
    Convert prior (mean, strength) to Beta(α, β) parameters.

    Args:
        mean: Prior mean ∈ (0, 1)
        strength: Prior strength λ (effective sample size)
            - λ = 1: Weak prior (Jeffreys-like)
            - λ = 10: Moderate prior
            - λ = 100: Strong prior

    Returns:
        Tuple of (alpha, beta) for Beta(α, β)

    Example:
        >>> alpha, beta = beta_prior_params(mean=0.3, strength=10)
        >>> # Returns (3.0, 7.0): prior centered at 0.3 with strength 10
    """
    if not (0 < mean < 1):
        raise ValueError(f"mean must be in (0, 1), got {mean}")
    if strength <= 0:
        raise ValueError(f"strength must be positive, got {strength}")

    alpha = mean * strength
    beta = (1 - mean) * strength
    return alpha, beta


def effective_sample_size(
    alpha_post: np.ndarray | float,
    beta_post: np.ndarray | float,
    alpha_prior: np.ndarray | float,
    beta_prior: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute effective sample size (ESS) contributed by data.

    ESS = (α_post + β_post) - (α_prior + β_prior)

    This measures how much information the data contributed beyond the prior.

    Args:
        alpha_post: Posterior α
        beta_post: Posterior β
        alpha_prior: Prior α
        beta_prior: Prior β

    Returns:
        Effective sample size (number of observations worth of information)

    Example:
        >>> # Prior: Beta(1, 1), Data: 5 successes, 3 failures
        >>> alpha_post, beta_post = 6, 4
        >>> alpha_prior, beta_prior = 1, 1
        >>> effective_sample_size(6, 4, 1, 1)
        8.0  # Equivalent to 8 observations
    """
    return (
        np.asarray(alpha_post)
        + np.asarray(beta_post)
        - np.asarray(alpha_prior)
        - np.asarray(beta_prior)
    )
