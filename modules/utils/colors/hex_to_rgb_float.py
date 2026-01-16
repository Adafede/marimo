"""Convert hex color to RGB float tuple."""

__all__ = ["hex_to_rgb_float"]

from typing import Tuple


def hex_to_rgb_float(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB float tuple (0.0-1.0 range)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)
