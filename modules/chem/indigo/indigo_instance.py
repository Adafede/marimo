"""Shared Indigo instance and renderer for molecule loading and rendering."""

__all__ = ["get_indigo", "get_renderer",]

from indigo import Indigo
from indigo.renderer import IndigoRenderer

# Single Indigo instance per process (required)
INDIGO = Indigo()
RENDERER = IndigoRenderer(INDIGO)

# Expose for other modules
def get_indigo():
    return INDIGO

def get_renderer():
    return RENDERER
