"""Beta distribution utilities for Bayesian inference.

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
    """Compute P(θ > threshold | data) for Beta(α, β) posterior.

                Uses the regularized incomplete beta function:
                    P(θ > t) = 1 - I_t(α, β)

                where I_t is the regularized incomplete beta function.

    Parameters
    ----------
    alpha : np.ndarray | float
        Alpha.
    beta : np.ndarray | float
        Beta.
    threshold : np.ndarray | float
        Threshold.

    Returns
    -------
    np.ndarray | float
        Posterior probabilities that ``theta`` is greater than ``threshold``.

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
    """Compute P(θ < threshold | data) for Beta(α, β) posterior.

    Parameters
    ----------
    alpha : np.ndarray | float
        Alpha.
    beta : np.ndarray | float
        Beta.
    threshold : np.ndarray | float
        Threshold.

    Returns
    -------
    np.ndarray | float
        Posterior probabilities that ``theta`` is less than ``threshold``.

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
    """Compute equal-tailed credible interval for Beta(α, β).

                Returns the interval [L, U] such that:
                    P(θ < L) = (1 - probability) / 2
                    P(θ > U) = (1 - probability) / 2

                For symmetric distributions, this equals the HDI (highest density interval).

    Parameters
    ----------
    alpha : np.ndarray | float
        Alpha.
    beta : np.ndarray | float
        Beta.
    probability : float
        Default is 0.89.

    Returns
    -------
    tuple[np.ndarray | float, np.ndarray | float]
        Lower and upper bounds of the equal-tailed credible interval.

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
    """Compute posterior mean E[θ] = α / (α + β).

    Parameters
    ----------
    alpha : np.ndarray | float
        Alpha.
    beta : np.ndarray | float
        Beta.

    Returns
    -------
    np.ndarray | float
        Posterior mean values.

    """
    a = np.asarray(alpha)
    b = np.asarray(beta)
    return a / (a + b)


def posterior_mode(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
) -> np.ndarray | float:
    """Compute posterior mode (MAP estimate).

                Mode = (α - 1) / (α + β - 2) when α > 1 and β > 1.
                Otherwise returns the mean as a fallback.

    Parameters
    ----------
    alpha : np.ndarray | float
        Alpha.
    beta : np.ndarray | float
        Beta.

    Returns
    -------
    np.ndarray | float
        Posterior mode (MAP) estimates.

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
    """Compute posterior variance Var[θ] = αβ / [(α+β)²(α+β+1)].

    Parameters
    ----------
    alpha : np.ndarray | float
        Alpha.
    beta : np.ndarray | float
        Beta.

    Returns
    -------
    np.ndarray | float
        Posterior variance values.

    """
    a = np.asarray(alpha)
    b = np.asarray(beta)
    ab_sum = a + b
    return (a * b) / (ab_sum**2 * (ab_sum + 1))


def beta_prior_params(
    mean: float,
    strength: float,
) -> tuple[float, float]:
    """Convert prior (mean, strength) to Beta(α, β) parameters.

    Parameters
    ----------
    mean : float
        Mean.
    strength : float
        Strength.

    Returns
    -------
    tuple[float, float]
        Prior parameters ``(alpha, beta)`` for the Beta distribution.

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
    """Compute effective sample size (ESS) contributed by data.

                ESS = (α_post + β_post) - (α_prior + β_prior)

                This measures how much information the data contributed beyond the prior.

    Parameters
    ----------
    alpha_post : np.ndarray | float
        Alpha post.
    beta_post : np.ndarray | float
        Beta post.
    alpha_prior : np.ndarray | float
        Alpha prior.
    beta_prior : np.ndarray | float
        Beta prior.

    Returns
    -------
    np.ndarray | float
        Effective sample size contributed by the observed data.

    """
    return (
        np.asarray(alpha_post)
        + np.asarray(beta_post)
        - np.asarray(alpha_prior)
        - np.asarray(beta_prior)
    )
