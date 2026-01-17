"""Wikidata RDF namespaces for rdflib."""

__all__ = ["WIKIDATA_NAMESPACES", "bind_wikidata_namespaces"]

from rdflib import Graph, Namespace

# Wikidata base URLs
ENTITY_BASE = "http://www.wikidata.org/entity/"
STATEMENT_BASE = "http://www.wikidata.org/entity/statement/"

# RDF Namespaces for Wikidata
WIKIDATA_NAMESPACES = {
    "WD": Namespace(ENTITY_BASE),
    "WDREF": Namespace(ENTITY_BASE.replace("entity/", "reference/")),
    "WDS": Namespace(STATEMENT_BASE),
    "WDT": Namespace(ENTITY_BASE.replace("entity/", "prop/direct/")),
    "P": Namespace(ENTITY_BASE.replace("entity/", "prop/")),
    "PS": Namespace(ENTITY_BASE.replace("entity/", "prop/statement/")),
    "PR": Namespace(ENTITY_BASE.replace("entity/", "prop/reference/")),
    "PROV": Namespace("http://www.w3.org/ns/prov#"),
    "SCHEMA": Namespace("http://schema.org/"),
}


def bind_wikidata_namespaces(graph: Graph) -> None:
    """Bind all Wikidata namespaces to an rdflib Graph."""
    for prefix, ns in WIKIDATA_NAMESPACES.items():
        graph.bind(prefix.lower(), ns)
