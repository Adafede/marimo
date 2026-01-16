"""Wrap image HTML in mo.image."""

__all__ = ["wrap_image"]

import re
from typing import Any

import marimo as mo


def wrap_image(html_str: str) -> Any:
    """Wrap image HTML in mo.image for mo.ui.table."""
    if not html_str:
        return mo.Html("")
    
    match = re.search(r'src="([^"]+)"', html_str)
    if match:
        return mo.image(src=match.group(1), width=150, height=100, rounded=True)
    return mo.Html(html_str)
