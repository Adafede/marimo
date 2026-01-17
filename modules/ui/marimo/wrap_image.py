"""Wrap image HTML for mo.ui.table display."""

__all__ = ["wrap_image"]

import re

import marimo as mo

# Pre-compiled regex for performance
SRC_PATTERN = re.compile(r'src="([^"]+)"')


def wrap_image(
    html_str: str,
    max_width: str = "150px",
    max_height: str = "100px",
    rounded: bool = True,
) -> mo.Html:
    """
    Wrap image HTML in mo.Html for mo.ui.table.

    Takes an HTML img tag string and returns it wrapped in mo.Html
    with consistent styling for table display.
    """
    if not html_str:
        return mo.Html("")

    # If it's already an img tag, extract src and rebuild with consistent styling
    match = SRC_PATTERN.search(html_str)
    if match:
        src = match.group(1)
        border_radius = "border-radius: 8px;" if rounded else ""
        styled_img = f'<img src="{src}" loading="lazy" style="max-width: {max_width}; max-height: {max_height}; {border_radius}" />'
        return mo.Html(styled_img)

    # If no src found, return as-is
    return mo.Html(html_str)
