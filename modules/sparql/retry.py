"""
Retry logic for network requests - no external dependencies.

Generic retry utilities that can wrap any callable.
"""

__all__ = [
    "with_retry",
    "RetryConfig",
]

import time
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    backoff_base: float = 2.0
    backoff_multiplier: float = 1.0

    def wait_time(self, attempt: int) -> float:
        """Calculate wait time for given attempt (0-indexed)."""
        return self.backoff_multiplier * (self.backoff_base**attempt)


def with_retry(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """
    Execute a function with retry logic.

    Args:
        func: Zero-argument callable to execute
        config: Retry configuration (defaults to 3 attempts, exponential backoff)
        on_retry: Optional callback called before each retry with (attempt, exception)

    Returns:
        Result of func()

    Raises:
        The last exception if all retries fail

    Example:
        >>> def flaky_request():
        ...     # might fail
        ...     return requests.get(url)
        >>> result = with_retry(flaky_request, RetryConfig(max_attempts=5))
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < config.max_attempts - 1:
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(config.wait_time(attempt))

    raise last_exception  # type: ignore
