"""Add resource to RDF graph."""

__all__ = ["add_resource"]

from rdflib import Graph, URIRef


def add_resource(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    uri: str | None,
) -> None:
    """Add a resource (URI) to the graph if uri is not None/empty."""
    if uri is not None and uri != "":
        graph.add((subject, predicate, URIRef(uri)))
