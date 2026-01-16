"""Wrap image HTML in mo.image."""

__all__ = ["wrap_image"]

import re

import marimo as mo


def wrap_image(
    html_str: str,
    width: int = 150,
    height: int = 100,
    rounded: bool = True,
) -> mo.Html | mo.image:
    """Wrap image HTML in mo.image for mo.ui.table."""
    if not html_str:
        return mo.Html("")
    
    match = re.search(r'src="([^"]+)"', html_str)
    if match:
        return mo.image(src=match.group(1), width=width, height=height, rounded=rounded)
    return mo.Html(html_str)
