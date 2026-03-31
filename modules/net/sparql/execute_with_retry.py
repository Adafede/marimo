"""Execute SPARQL query with retry logic."""

__all__ = ["execute_with_retry"]

import time
from .client import Client


def execute_with_retry(
    query: str,
    endpoint: str,
    timeout: int = 120,
    max_retries: int = 2,
    backoff_base: float = 1.0,
    format: str = "csv",
    fallback_endpoint: str | None = None,
) -> bytes:
    """Execute SPARQL query with retry logic.

    Optimized for fast feedback with minimal retries.
    """
    if not query or not query.strip():
        raise ValueError("SPARQL query cannot be empty")

    last_error = None
    last_http_details = None

    # Create clients once, reuse
    main_client = Client(endpoint=endpoint, timeout=timeout)
    fallback_client = (
        Client(endpoint=fallback_endpoint, timeout=timeout)
        if fallback_endpoint
        else None
    )

    for attempt in range(max_retries):
        # Use fallback endpoint on last attempt if provided
        client = (
            fallback_client
            if (attempt == max_retries - 1 and fallback_client)
            else main_client
        )

        try:
            if format == "json":
                return client.query_json(query=query)
            return client.query_csv(query=query)

        except Exception as e:
            last_error = e

            status = getattr(e, "code", None)
            if status is not None:
                body_preview = ""
                if hasattr(e, "read"):
                    try:
                        body_preview = e.read().decode("utf-8", errors="replace")[:400]  # ty: ignore[call-non-callable]
                    except Exception:
                        body_preview = ""
                last_http_details = (status, body_preview)

            # Don't sleep after the last attempt
            if attempt < max_retries - 1:
                sleep_time = backoff_base * (2**attempt)
                time.sleep(sleep_time)

    # All retries failed - raise appropriate error
    error_name = type(last_error).__name__
    error_msg = str(last_error)

    if "timeout" in error_name.lower() or "timeout" in error_msg.lower():
        raise TimeoutError(
            f"Query timed out after {max_retries} attempts.",
        ) from last_error
    elif "http" in error_name.lower() or "urlerror" in error_name.lower():
        if last_http_details is not None:
            status, body_preview = last_http_details
            detail = f"HTTP status {status}"
            if body_preview:
                detail = f"{detail}; response: {body_preview}"
            raise ConnectionError(f"HTTP error: {detail[:600]}") from last_error
        raise ConnectionError(f"HTTP error: {error_msg[:200]}") from last_error
    else:
        raise RuntimeError(f"{error_name}: {last_error}") from last_error
