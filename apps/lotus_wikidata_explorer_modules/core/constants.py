"""
Constants: RDF namespaces, URLs, SPARQL fragments, and other immutable values.
"""

import re
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS, XSD, DCTERMS

# ====================================================================
# CENTRALIZED RDF NAMESPACES AND URLS
# ====================================================================

# Namespaces (used throughout for RDF export)
WD = Namespace("http://www.wikidata.org/entity/")
WDREF = Namespace("http://www.wikidata.org/reference/")
WDS = Namespace("http://www.wikidata.org/entity/statement/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
P = Namespace("http://www.wikidata.org/prop/")
PS = Namespace("http://www.wikidata.org/prop/statement/")
PR = Namespace("http://www.wikidata.org/prop/reference/")
PROV = Namespace("http://www.w3.org/ns/prov#")
SCHEMA = Namespace("http://schema.org/")

# URLs (constants)
SCHOLIA_URL = "https://scholia.toolforge.org/"
WIKIDATA_URL = "https://www.wikidata.org/"
WIKIDATA_HTTP_URL = WIKIDATA_URL.replace("https://", "http://")
WIKIDATA_ENTITY_URL = WIKIDATA_HTTP_URL + "entity/"
WIKIDATA_WIKI_URL = WIKIDATA_URL + "wiki/"

# ====================================================================
# SPARQL QUERY FRAGMENTS
# ====================================================================

# Common SPARQL prefixes
SPARQL_PREFIXES = """
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX pr: <http://www.wikidata.org/prop/reference/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
"""

SACHEM_PREFIXES = """
PREFIX sachem: <http://bioinfo.uochb.cas.cz/rdf/v1.0/sachem#>
PREFIX idsm: <https://idsm.elixir-czech.cz/sparql/endpoint/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Common SELECT clause for compound queries
COMPOUND_SELECT_VARS = """
?compound ?compoundLabel ?compound_inchikey ?compound_smiles_conn
?compound_smiles_iso ?compound_mass ?compound_formula
?taxon_name ?taxon
?ref_qid ?ref_title ?ref_doi ?ref_date
?statement ?ref
"""

COMPOUND_MINIMAL_VARS = """
?compound ?compoundLabel ?compound_inchikey ?compound_smiles_conn
"""

COMPOUND_INTERIM_VARS = (
    COMPOUND_MINIMAL_VARS
    + """
?taxon ?taxon_name ?ref_qid ?statement ?ref
"""
)

# Common compound identifier retrieval (used in subqueries)
COMPOUND_IDENTIFIERS = """
?compound wdt:P235 ?compound_inchikey;
          wdt:P233 ?compound_smiles_conn.
"""

# Common taxon-reference association pattern (used in subqueries)
TAXON_REFERENCE_ASSOCIATION = """
?compound p:P703 ?statement.
?statement ps:P703 ?taxon;
           prov:wasDerivedFrom ?ref.
?ref pr:P248 ?ref_qid.
?taxon wdt:P225 ?taxon_name.
"""

# Common compound property optionals
COMPOUND_PROPERTIES_OPTIONAL = """
OPTIONAL { ?compound wdt:P2017 ?compound_smiles_iso. }
OPTIONAL { ?compound wdt:P2067 ?compound_mass. }
OPTIONAL { ?compound wdt:P274 ?compound_formula. }
OPTIONAL {
    ?compound rdfs:label ?compoundLabel.
    FILTER(LANG(?compoundLabel) = "en")
}
OPTIONAL {
    ?compound rdfs:label ?compoundLabel.
    FILTER(LANG(?compoundLabel) = "mul")
}
"""

# Common taxonomic and reference optionals
TAXONOMIC_REFERENCE_OPTIONAL = """
OPTIONAL {
    ?statement ps:P703 ?taxon;
               prov:wasDerivedFrom ?ref.
    ?ref pr:P248 ?ref_qid.
    ?compound p:P703 ?statement.
    ?taxon wdt:P225 ?taxon_name.
    OPTIONAL { ?ref_qid wdt:P1476 ?ref_title. }
    OPTIONAL { ?ref_qid wdt:P356 ?ref_doi. }
    OPTIONAL { ?ref_qid wdt:P577 ?ref_date. }
}
"""

# Common reference metadata retrieval (after subquery)
REFERENCE_METADATA_OPTIONAL = """
OPTIONAL { ?ref_qid wdt:P1476 ?ref_title. }
OPTIONAL { ?ref_qid wdt:P356 ?ref_doi. }
OPTIONAL { ?ref_qid wdt:P577 ?ref_date. }
"""

# ====================================================================
# TRANSLATION MAPS AND PATTERNS
# ====================================================================

# Subscript translation map (constant for performance)
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# Regex pattern for molecular formula parsing (compiled once)
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")

# Pluralization map (constant)
PLURAL_MAP = {
    "Entry": "Entries",
    "entry": "entries",
    "Taxon": "Taxa",
    "taxon": "taxa",
}
