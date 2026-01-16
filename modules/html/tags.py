"""
Pure HTML generation utilities - no external dependencies.

Functions that generate HTML strings. No framework dependencies.
"""

__all__ = [
    "link",
    "image",
    "styled_link",
]


def link(
    url: str,
    text: str,
    target: str = "_blank",
    rel: str = "noopener noreferrer",
) -> str:
    """
    Generate an HTML anchor tag.

    Args:
        url: The href URL
        text: Link text content
        target: Target attribute (default: "_blank")
        rel: Rel attribute for security

    Returns:
        HTML anchor tag string

    Example:
        >>> link("https://example.com", "Example")
        '<a href="https://example.com" target="_blank" rel="noopener noreferrer">Example</a>'
    """
    return f'<a href="{url}" target="{target}" rel="{rel}">{text}</a>'


def styled_link(
    url: str,
    text: str,
    color: str = "#3377c4",
    target: str = "_blank",
) -> str:
    """
    Generate a styled HTML anchor tag.

    Args:
        url: The href URL
        text: Link text content
        color: CSS color for the link
        target: Target attribute

    Returns:
        HTML anchor tag with inline style

    Example:
        >>> styled_link("https://example.com", "Example", "#ff0000")
        '<a href="https://example.com" target="_blank" style="color:#ff0000;">Example</a>'
    """
    return f'<a href="{url}" target="{target}" style="color:{color};">{text}</a>'


def image(
    src: str,
    alt: str = "",
    max_width: str = "200px",
    max_height: str = "150px",
    lazy: bool = True,
    rounded: bool = False,
) -> str:
    """
    Generate an HTML img tag.

    Args:
        src: Image source URL
        alt: Alt text
        max_width: CSS max-width
        max_height: CSS max-height
        lazy: Whether to use lazy loading
        rounded: Whether to add border-radius

    Returns:
        HTML img tag string

    Example:
        >>> image("img.png", "My Image", lazy=True)
        '<img src="img.png" alt="My Image" loading="lazy" style="max-width:200px;max-height:150px;" />'
    """
    parts = [f'<img src="{src}"']

    if alt:
        parts.append(f'alt="{alt}"')

    if lazy:
        parts.append('loading="lazy"')

    style_parts = [f"max-width:{max_width}", f"max-height:{max_height}"]
    if rounded:
        style_parts.append("border-radius:8px")

    parts.append(f'style="{";".join(style_parts)};"')
    parts.append("/>")

    return " ".join(parts)
