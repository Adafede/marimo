"""
RDF graph building utilities - requires rdflib.

Pure functions for constructing RDF graphs. No hidden configuration.
"""

__all__ = [
    "add_literal",
    "add_resource",
    "bind_namespaces",
]

from typing import Any, Optional, Dict

from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import XSD


def add_literal(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    value: Any,
    datatype=XSD.string,
    skip_empty: bool = True,
) -> None:
    """
    Add a literal triple to graph if value is non-empty.

    Args:
        graph: RDF graph to add to (mutated in place)
        subject: Subject URI
        predicate: Predicate URI
        value: Value to add
        datatype: XSD datatype for the literal
        skip_empty: If True, skip None and empty string values

    Example:
        >>> g = Graph()
        >>> add_literal(g, URIRef("http://example.org/item"),
        ...             URIRef("http://example.org/name"), "Test")
    """
    if skip_empty and (value is None or value == ""):
        return
    graph.add((subject, predicate, Literal(value, datatype=datatype)))


def add_resource(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    obj: URIRef,
) -> None:
    """
    Add a resource triple to graph.

    Args:
        graph: RDF graph to add to (mutated in place)
        subject: Subject URI
        predicate: Predicate URI
        obj: Object URI
    """
    graph.add((subject, predicate, obj))


def bind_namespaces(
    graph: Graph,
    namespaces: Dict[str, str],
) -> None:
    """
    Bind namespace prefixes to graph.

    Args:
        graph: RDF graph (mutated in place)
        namespaces: Dict mapping prefix to namespace URI

    Example:
        >>> g = Graph()
        >>> bind_namespaces(g, {"wd": "http://www.wikidata.org/entity/"})
    """
    for prefix, uri in namespaces.items():
        graph.bind(prefix, Namespace(uri))
