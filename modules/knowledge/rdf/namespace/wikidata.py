"""Wikidata RDF namespaces."""

__all__ = ["WIKIDATA_NAMESPACES", "WIKIDATA_PREFIXES"]

# Wikidata base URLs
ENTITY_BASE = "http://www.wikidata.org/entity/"
STATEMENT_BASE = "http://www.wikidata.org/entity/statement/"

# RDF Namespaces for Wikidata (as plain strings for maplib compatibility)
WIKIDATA_NAMESPACES = {
    "wd": ENTITY_BASE,
    "wdref": ENTITY_BASE.replace("entity/", "reference/"),
    "wds": STATEMENT_BASE,
    "wdt": ENTITY_BASE.replace("entity/", "prop/direct/"),
    "p": ENTITY_BASE.replace("entity/", "prop/"),
    "ps": ENTITY_BASE.replace("entity/", "prop/statement/"),
    "pr": ENTITY_BASE.replace("entity/", "prop/reference/"),
    "prov": "http://www.w3.org/ns/prov#",
    "schema": "http://schema.org/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "dcterms": "http://purl.org/dc/terms/",
}

# For backwards compatibility with uppercase keys
WIKIDATA_PREFIXES = {k.upper(): v for k, v in WIKIDATA_NAMESPACES.items()}
