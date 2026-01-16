"""Bind namespaces to RDF graph."""

__all__ = ["bind_namespaces"]

from rdflib import Graph, Namespace
from typing import Dict


def bind_namespaces(graph: Graph, namespaces: Dict[str, Namespace]) -> None:
    """Bind namespace prefixes to the graph."""
    for prefix, namespace in namespaces.items():
        graph.bind(prefix, namespace)
