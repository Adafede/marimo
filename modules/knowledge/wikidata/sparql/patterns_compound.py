"""SPARQL patterns for compound queries."""

__all__ = [
    "SELECT_VARS_FULL",
    "SELECT_VARS_MINIMAL",
    "SELECT_VARS_INTERIM",
    "IDENTIFIERS",
    "TAXON_REFERENCE_ASSOCIATION",
    "PROPERTIES_OPTIONAL",
    "TAXONOMIC_REFERENCE_OPTIONAL",
    "REFERENCE_METADATA_OPTIONAL",
]

SELECT_VARS_FULL = """
?compound ?compoundLabel ?compound_inchikey ?compound_smiles_conn
?compound_smiles_iso ?compound_mass ?compound_formula
?taxon_name ?taxon
?ref_qid ?ref_title ?ref_doi ?ref_date
?statement ?ref
"""

SELECT_VARS_MINIMAL = """
?compound ?compoundLabel ?compound_inchikey ?compound_smiles_conn
"""

SELECT_VARS_INTERIM = SELECT_VARS_MINIMAL + """
?taxon ?taxon_name ?ref_qid ?statement ?ref
"""

IDENTIFIERS = """
?compound wdt:P235 ?compound_inchikey;
          wdt:P233 ?compound_smiles_conn.
"""

TAXON_REFERENCE_ASSOCIATION = """
?compound p:P703 ?statement.
?statement ps:P703 ?taxon;
           prov:wasDerivedFrom ?ref.
?ref pr:P248 ?ref_qid.
?taxon wdt:P225 ?taxon_name.
"""

PROPERTIES_OPTIONAL = """
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

REFERENCE_METADATA_OPTIONAL = """
OPTIONAL { ?ref_qid wdt:P1476 ?ref_title. }
OPTIONAL { ?ref_qid wdt:P356 ?ref_doi. }
OPTIONAL { ?ref_qid wdt:P577 ?ref_date. }
"""
