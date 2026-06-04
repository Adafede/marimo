"""SPARQL patterns for compound queries."""

__all__ = [
    "SELECT_VARS_FULL",
    "SELECT_VARS_MINIMAL",
    "SELECT_VARS_INTERIM",
    "IDENTIFIERS",
    "COMPOUND_IDENTIFIERS",
    "TAXON_REFERENCE_ASSOCIATION",
    "PROPERTIES_OPTIONAL",
    "TAXONOMIC_REFERENCE_OPTIONAL",
    "REFERENCE_METADATA_OPTIONAL",
]

SELECT_VARS_FULL = """
(xsd:integer(STRAFTER(STR(?c), "Q")) AS ?compound)
?compoundLabel
?compound_inchikey
?compound_smiles_conn
?compound_smiles_iso
?compound_mass
?compound_formula
(xsd:integer(STRAFTER(STR(?t), "Q")) AS ?taxon)
?taxon_name
(xsd:integer(STRAFTER(STR(?r), "Q")) AS ?ref_qid)
?ref
?ref_title
?ref_doi
?ref_date
?statement
"""

SELECT_VARS_MINIMAL = """
?c
?compoundLabel
?compound_inchikey
?compound_smiles_conn
"""

SELECT_VARS_INTERIM = (
    SELECT_VARS_MINIMAL
    + """
?t
?taxon_name
?r
?ref
?statement
"""
)

IDENTIFIERS = """
?c wdt:P235 ?compound_inchikey;
          wdt:P233 ?compound_smiles_conn.
"""

# Alias for clearer semantics
COMPOUND_IDENTIFIERS = IDENTIFIERS

TAXON_REFERENCE_ASSOCIATION = """
?c p:P703 ?statement.
?statement ps:P703 ?t;
           prov:wasDerivedFrom ?ref.
?ref pr:P248 ?r.
?t wdt:P225 ?taxon_name.
"""

PROPERTIES_OPTIONAL = """
OPTIONAL { ?c wdt:P2017 ?compound_smiles_iso. }
OPTIONAL { ?c wdt:P2067 ?compound_mass. }
OPTIONAL { ?c wdt:P274 ?compound_formula. }
OPTIONAL {
    ?c rdfs:label ?compoundLabel.
    FILTER(LANG(?compoundLabel) = "en")
}
OPTIONAL {
    ?c rdfs:label ?compoundLabel.
    FILTER(LANG(?compoundLabel) = "mul")
}
"""

TAXONOMIC_REFERENCE_OPTIONAL = """
OPTIONAL {
    ?statement ps:P703 ?t;
               prov:wasDerivedFrom ?ref.
    ?ref pr:P248 ?r.
    ?c p:P703 ?statement.
    ?t wdt:P225 ?taxon_name.
    OPTIONAL { ?r wdt:P1476 ?ref_title. }
    OPTIONAL { ?r wdt:P356 ?ref_doi. }
    OPTIONAL { ?r wdt:P577 ?ref_date. }
}
"""

REFERENCE_METADATA_OPTIONAL = """
OPTIONAL { ?r wdt:P1476 ?ref_title. }
OPTIONAL { ?r wdt:P356 ?ref_doi. }
OPTIONAL { ?r wdt:P577 ?ref_date. }
"""
