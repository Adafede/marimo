"""
Common RDF namespace definitions.

Pre-defined namespaces for convenience. Users can import these
or define their own.
"""

__all__ = [
    "WIKIDATA_NAMESPACES",
    "COMMON_NAMESPACES",
]

# Wikidata-specific namespaces
WIKIDATA_NAMESPACES = {
    "wd": "http://www.wikidata.org/entity/",
    "wds": "http://www.wikidata.org/entity/statement/",
    "wdref": "http://www.wikidata.org/reference/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "pr": "http://www.wikidata.org/prop/reference/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
}

# Common namespaces
COMMON_NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/",
    "dcterms": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

