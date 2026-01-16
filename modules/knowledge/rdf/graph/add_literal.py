"""Add literal to RDF graph."""

__all__ = ["add_literal"]

from rdflib import Graph, Literal, URIRef


def add_literal(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    value: str | None,
    datatype: URIRef | None = None,
) -> None:
    """Add a literal value to the graph if value is not None/empty."""
    if value is not None and value != "":
        literal = Literal(value, datatype=datatype) if datatype else Literal(value)
        graph.add((subject, predicate, literal))
