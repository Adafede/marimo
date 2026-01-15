# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "adafedemarimo==0.1.0",
#     "marimo",
# ]
#
#
# [tool.marimo.display]
# theme = "system"
# ///

"""
Example app demonstrating the refactored adafedemarimo package.

The package is organized into focused subpackages:
- text/     Pure text utilities (no dependencies)
- html/     HTML generation (no dependencies)
- sparql/   SPARQL client (urllib only)
- df/       DataFrame utilities (requires polars)
- rdf/      RDF utilities (requires rdflib)
- wikidata/ Wikidata utilities (no heavy deps)
"""

import marimo

__generated_with = "0.19.2"
app = marimo.App(app_title="Package Demo")

with app.setup:
    import marimo as mo

    from adafedemarimo.df.filters import filter_range
    from adafedemarimo.df.transforms import rename_columns

    from adafedemarimo.text import (
        validate_smiles,
        parse_formula,
        pluralize)
    from adafedemarimo.wikidata import extract_qid
    from adafedemarimo.html import structure_image_url, styled_link

    from adafedemarimo.sparql import SPARQLClient, with_retry, RetryConfig


@app.cell
def intro():
    mo.md("""
    # 🧪 adafedemarimo Package Demo
    
    ## Package Structure
    
    ```
    adafedemarimo/
    ├── text/           # Pure text utilities (NO dependencies)
    │   ├── strings.py  # pluralize, truncate
    │   ├── smiles.py   # validate_smiles, escape_for_sparql
    │   └── formula.py  # parse_formula, count_element
    │
    ├── html/           # HTML generation (NO dependencies)
    │   ├── tags.py     # link, image, styled_link
    │   └── urls.py     # build_query_string, structure_image_url
    │
    ├── sparql/         # SPARQL client (urllib only)
    │   ├── client.py   # SPARQLClient, query
    │   ├── retry.py    # with_retry, RetryConfig
    │   └── builders.py # values_clause, optional_block
    │
    ├── df/             # DataFrame ops (requires: polars)
    │   ├── filters.py  # filter_range, filter_by_values
    │   └── transforms.py # rename_columns, coalesce_columns
    │
    ├── rdf/            # RDF utilities (requires: rdflib)
    │   ├── graph.py    # add_literal, add_resource
    │   └── namespaces.py # WIKIDATA_NAMESPACES
    │
    └── wikidata/       # Wikidata utils (NO heavy deps)
        ├── qid.py      # extract_qid, is_qid, entity_url
        └── urls.py     # WIKIDATA_SPARQL_ENDPOINT, scholia_url
    ```
    
    ## Key Design Principles
    
    1. **Zero core dependencies** - Base package works without installing anything
    2. **Optional extras** - Install only what you need: `uv add adafedemarimo[df]`
    3. **Pure functions** - No hidden config, all params are explicit
    4. **Single responsibility** - Each module does ONE thing well
    """)
    return


@app.cell 
def test_text():
    # Test text utilities (zero dependencies!)
    results = []

    # SMILES validation
    valid, err = validate_smiles("c1ccccc1")
    results.append(f"✅ SMILES 'c1ccccc1' valid: {valid}")

    # Formula parsing
    atoms = parse_formula("C6H12O6")
    results.append(f"✅ Formula C6H12O6: {atoms}")

    # Pluralize with custom irregular
    taxa = pluralize("Taxon", 5, {"Taxon": "Taxa"})
    results.append(f"✅ Pluralize 'Taxon' x5: {taxa}")

    mo.md(f"""
    ## Text Utilities (no dependencies)
    
    {chr(10).join(results)}
    """)
    return


@app.cell
def test_html():
    # Test HTML utilities
    qid = extract_qid("http://www.wikidata.org/entity/Q2270")
    link_html = styled_link(f"https://scholia.toolforge.org/{qid}", qid, color="#006699")
    img_url = structure_image_url("c1ccccc1")

    mo.md(f"""
    ## HTML Utilities (no dependencies)
    
    - Extracted QID: `{qid}`
    - Styled link: {mo.Html(link_html)}
    - Structure URL: `{img_url[:60]}...`
    """)
    return


@app.cell
def test_sparql():
    # Test SPARQL utilities
    config = RetryConfig(max_attempts=3, backoff_base=2.0)
    client = SPARQLClient("https://query.wikidata.org/sparql", timeout=30)

    mo.md(f"""
    ## SPARQL Utilities (urllib only)
    
    - RetryConfig: {config.max_attempts} attempts, {config.backoff_base}x backoff
    - SPARQLClient endpoint: `{client.endpoint}`
    - SPARQLClient timeout: {client.timeout}s
    """)
    return


if __name__ == "__main__":
    app.run()
