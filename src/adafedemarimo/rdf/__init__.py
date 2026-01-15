"""
RDF utilities subpackage - requires rdflib.

Functions for building RDF graphs and common namespace definitions.
"""

from adafedemarimo.rdf.graph import (
    add_literal,
    add_resource,
    bind_namespaces,
)

from adafedemarimo.rdf.namespaces import (
    WIKIDATA_NAMESPACES,
    COMMON_NAMESPACES,
)

__all__ = [
    # graph
    "add_literal",
    "add_resource",
    "bind_namespaces",
    # namespaces
    "WIKIDATA_NAMESPACES",
    "COMMON_NAMESPACES",
]

