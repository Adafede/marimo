"""
SPARQL utilities subpackage.

Contains client, retry logic, and query building utilities.
"""

from adafedemarimo.sparql.client import (
    SPARQLClient,
    query,
)

from adafedemarimo.sparql.retry import (
    with_retry,
    RetryConfig,
)

from adafedemarimo.sparql.builders import (
    values_clause,
    optional_block,
    filter_lang,
    prefix_block,
)

__all__ = [
    # client
    "SPARQLClient",
    "query",
    # retry
    "with_retry",
    "RetryConfig",
    # builders
    "values_clause",
    "optional_block",
    "filter_lang",
    "prefix_block",
]

