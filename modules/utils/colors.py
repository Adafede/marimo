"""
Color conversion utilities - no external dependencies.

Pure functions for color format conversions (hex, RGB, etc.).
"""

__all__ = [
    "hex_to_rgb_float",
    "hex_to_rgb",
]

from typing import Tuple


def hex_to_rgb_float(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color to RGB float tuple (0.0-1.0).

    Args:
        hex_color: Hex color string (e.g., "#77aadd" or "77aadd")

    Returns:
        Tuple of (r, g, b) with values from 0.0 to 1.0

    Example:
        >>> hex_to_rgb_float("#ff8000")
        (1.0, 0.5019607843137255, 0.0)
        >>> hex_to_rgb_float("ffffff")
        (1.0, 1.0, 1.0)
    """
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB integer tuple (0-255).

    Args:
        hex_color: Hex color string (e.g., "#77aadd" or "77aadd")

    Returns:
        Tuple of (r, g, b) with values from 0 to 255

    Example:
        >>> hex_to_rgb("#ff8000")
        (255, 128, 0)
    """
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
