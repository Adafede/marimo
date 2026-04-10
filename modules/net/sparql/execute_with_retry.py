"""Execute SPARQL query with retry logic."""

__all__ = ["execute_with_retry"]

import time
import re
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

    def _extract_status(exc: Exception) -> int | None:
        for attr in ("code", "status"):
            value = getattr(exc, attr, None)
            if isinstance(value, int) and 100 <= value <= 599:
                return value

        match = re.search(
            r"(?:HTTP\s+status|HTTP\s+Error|status[=: ]+)\s*(\d{3})",
            str(exc),
            re.IGNORECASE,
        )
        if match:
            return int(match.group(1))
        return None

    def _detect_upstream_error_payload(payload: bytes) -> str | None:
        if not payload:
            return None

        sample = payload[:2048].decode("utf-8", errors="replace")
        sample_lower = sample.lower()

        html_markers = ("<html", "<!doctype", "<head", "<title")
        gateway_markers = (
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "upstream",
            "nginx",
            "cloudflare",
        )

        if any(marker in sample_lower for marker in html_markers) and any(
            marker in sample_lower for marker in gateway_markers
        ):
            status_match = re.search(r"\b(50[0-9])\b", sample)
            status_hint = f"HTTP status {status_match.group(1)}; " if status_match else ""
            cleaned = " ".join(sample.split())
            return f"{status_hint}response: {cleaned[:500]}"

        if re.search(
            r"\b(50[0-9])\s+(?:bad gateway|service unavailable|gateway timeout)\b",
            sample,
            re.IGNORECASE,
        ):
            cleaned = " ".join(sample.split())
            return f"response: {cleaned[:500]}"

        return None

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
            payload = client.query_csv(query=query)
            payload_error = _detect_upstream_error_payload(payload)
            if payload_error is not None:
                raise ConnectionError(f"HTTP error: {payload_error}")
            return payload

        except Exception as e:
            last_error = e

            status = _extract_status(e)
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
    elif (
        "http" in error_name.lower()
        or "urlerror" in error_name.lower()
        or "http" in error_msg.lower()
        or _extract_status(last_error) is not None
        or isinstance(last_error, ConnectionError)
    ):
        if last_http_details is not None:
            status, body_preview = last_http_details
            detail = f"HTTP status {status}"
            if body_preview:
                detail = f"{detail}; response: {body_preview}"
            raise ConnectionError(f"HTTP error: {detail[:600]}") from last_error
        raise ConnectionError(f"HTTP error: {error_msg[:200]}") from last_error
    else:
        raise RuntimeError(f"{error_name}: {last_error}") from last_error
