"""Add literal to RDF graph."""

__all__ = ["add_literal"]

from typing import Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD


def add_literal(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    value: Optional[str],
    datatype: Optional[URIRef] = None,
) -> None:
    """Add a literal value to the graph if value is not None/empty."""
    if value is not None and value != "":
        literal = Literal(value, datatype=datatype) if datatype else Literal(value)
        graph.add((subject, predicate, literal))
