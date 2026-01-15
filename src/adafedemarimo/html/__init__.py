"""
HTML utilities subpackage - no external dependencies.

Pure functions for HTML generation and URL building.
"""

from adafedemarimo.html.tags import (
    link,
    image,
    styled_link,
)

from adafedemarimo.html.urls import (
    build_query_string,
    structure_image_url,
    doi_url,
    scholia_url,
)

__all__ = [
    # tags
    "link",
    "image",
    "styled_link",
    # urls
    "build_query_string",
    "structure_image_url",
    "doi_url",
    "scholia_url",
]

