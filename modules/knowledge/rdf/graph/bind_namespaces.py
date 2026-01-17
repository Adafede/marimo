"""Bind namespaces to RDF graph."""

__all__ = ["bind_namespaces"]

from rdflib import Graph, Namespace


def bind_namespaces(graph: Graph, namespaces: dict[str, Namespace]) -> None:
    """Bind namespace prefixes to the graph."""
    for prefix, namespace in namespaces.items():
        graph.bind(prefix, namespace)
