# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars==1.35.1",
#     "sparqlwrapper==2.0.0",
#     "pyarrow>=14.0.0",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import polars as pl
    import time
    from datetime import datetime
    from SPARQLWrapper import SPARQLWrapper, JSON
    from urllib.parse import quote


@app.function
def execute_sparql(
    query: str, user_agent: str = "LOTUS Explorer/2.0", max_retries: int = 3
) -> dict:
    """Execute SPARQL query with retry logic"""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", user_agent)

    for attempt in range(max_retries):
        try:
            return sparql.query().convert()
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Query failed after {max_retries} attempts: {str(e)}")
            time.sleep(2**attempt)


@app.function
def extract_qid(url: str) -> str:
    """Extract QID from Wikidata URL"""
    return url.replace("http://www.wikidata.org/entity/", "")


@app.function
def create_structure_image_url(smiles: str) -> str:
    """Create CDK Depict image URL from SMILES"""
    return f"https://www.simolecule.com/cdkdepict/depict/bot/svg?smi={quote(smiles)}&annotate=cip"


@app.function
def taxon_name_to_qid(taxon_name: str) -> str | None:
    """Get Wikidata QID from taxon name"""
    query = f"""
    SELECT ?item WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{taxon_name}";
                        mwapi:language "mul".
        ?item wikibase:apiOutputItem mwapi:item.
        ?num wikibase:apiOrdinal true.
      }}
      ?item wdt:P225 ?search.
      FILTER (?num = 0)
    }}
    """

    try:
        results = execute_sparql(query)
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            return extract_qid(bindings[0].get("item", {}).get("value", ""))
    except Exception:
        pass
    return None


@app.function
def get_binding_value(binding: dict, key: str, default: str = "") -> str:
    """Safely extract value from SPARQL binding"""
    return binding.get(key, {}).get("value", default)


@app.function
def create_link(url, text):
    # Use a deep blue for light mode (good contrast against white)
    # Use a brighter, more saturated blue for dark mode (good contrast against dark grey)
    light_mode_color = "#0b68cb"  # Deep Blue
    dark_mode_color = "#479bf5"  # Brighter Blue

    # This CSS variable or function isn't natively supported in all browser contexts
    # so we'll apply a standard color for simplicity in the current implementation,
    # A standard, accessible blue like #007AFF works well across both.
    accessible_blue = "#007AFF"
    return mo.Html(
        f'<a href="{url}" target="_blank"><span style="color: {accessible_blue};">{text}</span></a>'
    )


@app.function
def query_wikidata(
    qid: str, year_start: int | None = None, year_end: int | None = None
) -> pl.DataFrame:
    """Query Wikidata for natural products"""
    query = f"""
    SELECT DISTINCT ?structure ?structureLabel ?inchikey ?smiles_iso ?smiles_conn
                   ?taxon_name ?taxon ?ref_title ?ref_doi ?ref_qid ?pub_date
    WHERE {{
      ?taxon (wdt:P171*) wd:{qid};
             wdt:P225 ?taxon_name. 
      ?structure wdt:P235 ?inchikey;
                 wdt:P233 ?smiles_conn;
                 p:P703 ?statement.
      ?statement ps:P703 ?taxon;
                 prov:wasDerivedFrom ?ref.
      ?ref pr:P248 ?ref_qid.
      OPTIONAL {{ ?structure wdt:P2017 ?smiles_iso. }}
      OPTIONAL {{
        SERVICE <https://query-scholarly.wikidata.org/sparql> {{
          ?ref_qid wdt:P1476 ?ref_title;
                   wdt:P356 ?ref_doi;
                   wdt:P577 ?pub_date.
        }}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 100000
    """

    results = execute_sparql(query)
    bindings = results.get("results", {}).get("bindings", [])

    if not bindings:
        return pl.DataFrame()

    rows = [
        {
            "structure": get_binding_value(b, "structure"),
            "name": get_binding_value(b, "structureLabel"),
            "inchikey": get_binding_value(b, "inchikey"),
            "smiles": get_binding_value(b, "smiles_iso")
            or get_binding_value(b, "smiles_conn"),
            "taxon_name": get_binding_value(b, "taxon_name"),
            "taxon": get_binding_value(b, "taxon"),
            "ref_title": get_binding_value(b, "ref_title"),
            "ref_doi": get_binding_value(b, "ref_doi"),
            "reference": get_binding_value(b, "ref_qid"),
            "pub_date": get_binding_value(b, "pub_date"),
        }
        for b in bindings
    ]

    df = pl.DataFrame(rows)

    # Apply year filter if specified
    if year_start is not None and year_end is not None and "pub_date" in df.columns:
        df = (
            df.with_columns(
                [
                    pl.col("pub_date")
                    .str.slice(0, 4)
                    .cast(pl.Int32, strict=False)
                    .alias("year")
                ]
            )
            .filter(
                pl.col("year").is_null()
                | ((pl.col("year") >= year_start) & (pl.col("year") <= year_end))
            )
            .drop("year")
        )

    # Remove duplicates and sort
    return df.unique(subset=["structure", "taxon", "reference"], keep="first").sort(
        "name"
    )


@app.cell
def _():
    mo.md("""
    # 🌿 LOTUS Wikidata Natural Products Explorer

    Explore natural products from [LOTUS](https://doi.org/10.7554/eLife.70780) and 
    [Wikidata](https://www.wikidata.org/) for any taxon.

    Enter a taxon name to discover all associated natural products and their biological sources.
    """)
    return


@app.cell
def _():
    taxon_input = mo.ui.text(
        value="Swertia",
        label="Taxon name",
        placeholder="e.g., Swertia, Artemisia, Cannabis",
    )

    current_year = datetime.now().year
    year_start = mo.ui.number(
        value=1900, start=1700, stop=current_year, label="Start year"
    )
    year_end = mo.ui.number(
        value=current_year, start=1700, stop=current_year, label="End year"
    )
    year_filter = mo.ui.checkbox(label="Filter by publication year", value=False)

    run_button = mo.ui.run_button(label="🔍 Search Wikidata")

    mo.vstack(
        [
            mo.md("## Search Parameters"),
            mo.hstack(
                [taxon_input, year_start, year_end, year_filter], gap=2, widths="equal"
            ),
            run_button,
        ]
    )
    return run_button, taxon_input, year_end, year_filter, year_start


@app.cell
def _(run_button, taxon_input, year_end, year_filter, year_start):
    if not run_button.value:
        results_df = None
        qid = None
    else:
        with mo.status.spinner(title=f"Querying Wikidata for {taxon_input.value}..."):
            qid = taxon_name_to_qid(taxon_input.value)

            if not qid:
                mo.stop(
                    True,
                    mo.md(
                        f"**Error:** Could not find '{taxon_input.value}' in Wikidata"
                    ),
                )

            try:
                y_start = year_start.value if year_filter.value else None
                y_end = year_end.value if year_filter.value else None
                results_df = query_wikidata(qid, y_start, y_end)
            except Exception as e:
                mo.stop(True, mo.md(f"**Error:** {str(e)}"))
    return qid, results_df


@app.cell
def _(qid, results_df, run_button, taxon_input):
    if not run_button.value or results_df is None:
        summary_display = mo.md("")
    elif len(results_df) == 0:
        summary_display = mo.callout(
            mo.md(
                f"No natural products found for **{taxon_input.value}** (QID: {qid})"
            ),
            kind="warn",
        )
    else:
        n_compounds = results_df.n_unique(subset=["structure"])
        n_taxa = results_df.n_unique(subset=["taxon"])
        n_refs = results_df.n_unique(subset=["reference"])

        summary_display = mo.vstack(
            [
                mo.md(f"""
            ## Results Summary

            Found data for **{taxon_input.value}** ([{qid}](https://www.wikidata.org/wiki/{qid}))
            """),
                mo.hstack(
                    [
                        mo.stat(
                            value=str(n_compounds), label="Compounds", bordered=True
                        ),
                        mo.stat(value=str(n_taxa), label="Taxa", bordered=True),
                        mo.stat(value=str(n_refs), label="References", bordered=True),
                        mo.stat(
                            value=str(len(results_df)),
                            label="Total Entries",
                            bordered=True,
                        ),
                    ],
                    gap=2,
                ),
            ]
        )

    summary_display
    return


@app.cell
def _(results_df, run_button):
    if run_button.value and results_df is not None and len(results_df) > 0:
        search_box = mo.ui.text(
            label="Filter compounds",
            placeholder="Search by name (e.g., 'acid', 'glycoside')...",
        )
        mo.md(f"## 🔬 Compound Browser\n\n{search_box}")
    else:
        search_box = None
    return (search_box,)


@app.cell
def _(results_df, run_button, search_box):
    if not run_button.value or results_df is None or len(results_df) == 0:
        filtered_df = None
    else:
        filtered_df = results_df
        if search_box and search_box.value:
            filtered_df = filtered_df.filter(
                pl.col("name").str.to_lowercase().str.contains(search_box.value.lower())
            )
    return (filtered_df,)


@app.cell
def _(filtered_df, run_button):
    if not run_button.value or filtered_df is None or len(filtered_df) == 0:
        table_output = None
    else:
        # Create display table with images (limited for performance)
        display_data = []
        for _row in filtered_df.iter_rows(named=True):
            _img_url = create_structure_image_url(_row["smiles"])
            _struct_qid = extract_qid(_row["structure"])
            _taxon_qid = extract_qid(_row["taxon"])
            _reference_qid = extract_qid(_row["reference"])

            display_data.append(
                {
                    "2D Depiction": mo.image(src=_img_url),
                    "Compound name": _row["name"],
                    "Compound SMILES": _row["smiles"],
                    "Compound InChIKey": _row["inchikey"],
                    "Taxon name": _row["taxon_name"],
                    "Reference title": _row["ref_title"][:50] + "..."
                    if _row["ref_title"] and len(_row["ref_title"]) > 50
                    else _row["ref_title"],
                    "Reference DOI": create_link(
                        f'https://doi.org/{_row["ref_doi"]}', _row["ref_doi"]
                    )
                    if _row["ref_doi"]
                    else "",
                    "Compound QID": create_link(_row["structure"], _struct_qid),
                    "Taxon QID": create_link(_row["taxon"], _taxon_qid),
                    "Reference QID": create_link(_row["reference"], _reference_qid),
                }
            )

        display_table = mo.ui.table(display_data, selection=None, page_size=10)

        # Create export table (all data, CSV-ready) - use native Polars operations
        export_df = filtered_df.with_columns(
            [
                pl.col("structure")
                .str.replace("http://www.wikidata.org/entity/", "", literal=True)
                .alias("compound_qid"),
                pl.col("taxon")
                .str.replace("http://www.wikidata.org/entity/", "", literal=True)
                .alias("taxon_qid"),
                pl.col("reference")
                .str.replace("http://www.wikidata.org/entity/", "", literal=True)
                .alias("reference_qid"),
            ]
        ).select(
            [
                pl.col("name").alias("compound_name"),
                pl.col("smiles").alias("compound_smiles"),
                pl.col("inchikey").alias("compound_inchikey"),
                pl.col("taxon_name").alias("taxon_name"),
                pl.col("ref_title").alias("reference_title"),
                pl.col("ref_doi").alias("reference_doi"),
                "compound_qid",
                "taxon_qid",
                "reference_qid",
            ]
        )

        export_table = mo.ui.table(
            export_df,
            selection=None,
            page_size=20,
            style_cell=lambda row, col, selected: {
                "white-space": "nowrap",
                "overflow": "hidden",
                "text-overflow": "ellipsis",
                "max-width": "200px",
            },
        )

        table_output = mo.vstack(
            [
                mo.md(f"""
            ## Compounds Table
            """),
                mo.ui.tabs({"Display": display_table, "Export": export_table}),
            ]
        )

    table_output
    return


@app.cell
def _():
    mo.md("""
    ---
    **Data:** [LOTUS](https://www.wikidata.org/wiki/Q104225190) & [Wikidata](https://www.wikidata.org/) ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)) | 
    **Structure Rendering:** [CDK Depict](https://www.simolecule.com/cdkdepict/depict.html) | 
    **Code:** [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html) | 
    **Built with:** [marimo](https://marimo.io/)
    """)
    return


if __name__ == "__main__":
    app.run()
