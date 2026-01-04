"""
SPARQL query executor with retry logic and error handling.
"""

from functools import lru_cache
from typing import Dict, Any
import time

try:
    import httpx
    from sparqlx import SPARQLWrapper
except ImportError:
    httpx = None
    SPARQLWrapper = None

from ..core.config import CONFIG

__all__ = ["execute_sparql", "get_sparql_wrapper"]

# Global SPARQL wrapper for connection reuse (significant performance improvement)
_SPARQL_WRAPPER = None


def get_sparql_wrapper() -> "SPARQLWrapper":
    """
    Get or create the global SPARQL wrapper instance.

    Uses connection pooling for better performance on repeated queries.
    """
    global _SPARQL_WRAPPER
    if _SPARQL_WRAPPER is None:
        if SPARQLWrapper is None:
            raise ImportError("sparqlx is required for SPARQL queries")

        # Create httpx client with connection pooling
        if httpx:
            limits = httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            )
            client = httpx.Client(
                timeout=CONFIG["query_timeout"],
                limits=limits,
                follow_redirects=True,
            )
            _SPARQL_WRAPPER = SPARQLWrapper(
                sparql_endpoint=CONFIG["sparql_endpoint"],
                client=client,
            )
        else:
            _SPARQL_WRAPPER = SPARQLWrapper(
                sparql_endpoint=CONFIG["sparql_endpoint"],
                client_config={"timeout": CONFIG["query_timeout"]},
            )
    return _SPARQL_WRAPPER


@lru_cache(maxsize=128)
def execute_sparql(
    query: str, max_retries: int = CONFIG["max_retries"]
) -> Dict[str, Any]:
    """
    Execute SPARQL query using sparqlx with comprehensive error handling.

    Args:
        query: SPARQL query string
        max_retries: Maximum number of retry attempts

    Returns:
        JSON response from SPARQL endpoint

    Raises:
        ValueError: If query is empty or response is invalid
        TimeoutError: If query times out after max retries
        RuntimeError: For other SPARQL execution errors
    """
    if not query or not query.strip():
        raise ValueError("SPARQL query cannot be empty")

    if httpx is None:
        raise ImportError("httpx is required for SPARQL queries")

    sparql_wrapper = get_sparql_wrapper()

    for attempt in range(max_retries):
        try:
            # sparqlx handles the POST request with proper headers automatically
            response = sparql_wrapper.query(query, response_format="json")

            # Validate response structure
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(
                    f"Invalid SPARQL response: expected dict, got {type(result).__name__}. "
                    f"The endpoint may be experiencing issues."
                )

            # Ensure required structure exists
            if "results" not in result:
                raise ValueError(
                    "SPARQL response missing 'results' field. "
                    "The endpoint returned an unexpected format."
                )

            return result

        except httpx.TimeoutException as e:
            # Handle timeout exceptions specifically
            if attempt == max_retries - 1:
                raise TimeoutError(
                    f"⏱️ Query timed out after {max_retries} attempts "
                    f"({CONFIG['query_timeout']}s timeout each).\n\n"
                    f"💡 Suggestions:\n"
                    f"  • Query may be too complex - try narrowing search criteria\n"
                    f"  • SPARQL endpoint may be overloaded - try again later\n"
                    f"  • For large datasets, consider using the CLI with longer timeout\n\n"
                    f"Original error: {str(e)}"
                ) from e

            # Exponential backoff
            wait_time = CONFIG["retry_backoff"] ** attempt
            time.sleep(wait_time)

        except httpx.HTTPStatusError as e:
            # HTTP errors (4xx, 5xx)
            status_code = e.response.status_code

            if status_code == 429:  # Rate limiting
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"🚫 Rate limit exceeded after {max_retries} attempts. "
                        f"Please wait a few minutes before trying again."
                    ) from e
                # Exponential backoff for rate limits
                wait_time = CONFIG["retry_backoff"] ** (attempt + 2)
                time.sleep(wait_time)

            elif 500 <= status_code < 600:  # Server errors
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"❌ SPARQL endpoint error (HTTP {status_code}) after {max_retries} attempts.\n"
                        f"The endpoint may be temporarily unavailable. Please try again later."
                    ) from e
                # Retry with backoff
                wait_time = CONFIG["retry_backoff"] ** attempt
                time.sleep(wait_time)

            else:  # Other HTTP errors (e.g., 400 Bad Request)
                raise RuntimeError(
                    f"❌ SPARQL query failed with HTTP {status_code}.\n"
                    f"This usually indicates a malformed query or endpoint issue.\n\n"
                    f"Error details: {str(e)}"
                ) from e

        except httpx.NetworkError as e:
            # Network connectivity issues
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"🌐 Network error after {max_retries} attempts.\n"
                    f"Please check your internet connection and try again.\n\n"
                    f"Error: {str(e)}"
                ) from e
            wait_time = CONFIG["retry_backoff"] ** attempt
            time.sleep(wait_time)

        except (ValueError, KeyError) as e:
            # Validation errors - don't retry
            raise

        except Exception as e:
            # Unexpected errors
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"❌ Unexpected error during SPARQL query execution:\n{str(e)}"
                ) from e
            wait_time = CONFIG["retry_backoff"] ** attempt
            time.sleep(wait_time)

    # Should never reach here, but just in case
    raise RuntimeError(f"Failed to execute SPARQL query after {max_retries} attempts")
