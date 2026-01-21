"""Generate HTML <img> tag for an image."""

__all__ = ["html_from_image"]

DEFAULT_MAX_WIDTH: str = "150px"
DEFAULT_MAX_HEIGHT: str = "100px"
DEFAULT_BORDER_RADIUS: str = "8px"


def build_style(
    max_width: str,
    max_height: str,
    rounded: bool,
) -> str:
    """Build CSS style string for an <img> tag."""
    styles = {
        "max-width": max_width,
        "max-height": max_height,
    }
    if rounded:
        styles["border-radius"] = DEFAULT_BORDER_RADIUS
    # Join styles into a CSS string
    return "; ".join(f"{k}: {v}" for k, v in styles.items())


def html_from_image(
    img: str,
    max_width: str = DEFAULT_MAX_WIDTH,
    max_height: str = DEFAULT_MAX_HEIGHT,
    rounded: bool = True,
    lazy: bool = True,
) -> str:
    """
    Generate an HTML <img> tag for a chemical structure depiction or any image.

    Args:
        img: Image source URL or data URI
        max_width: Maximum width of the image (CSS)
        max_height: Maximum height of the image (CSS)
        rounded: Whether to apply border-radius
        lazy: Whether to add lazy loading attribute

    Returns:
        HTML <img> tag as a string. Returns empty string if img is empty.
    """
    if not img:
        return ""

    style = build_style(max_width=max_width, max_height=max_height, rounded=rounded)
    loading_attr = 'loading="lazy" ' if lazy else ""

    return f'<img src="{img}" {loading_attr}style="{style};" />'
