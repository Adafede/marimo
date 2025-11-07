# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "astral==3.2",
#     "polars==1.35.1",
#     "pyarrow==22.0.0",
#     "sparqlwrapper==2.0.0",
# ]
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import polars as pl
    import time
    from datetime import datetime
    from functools import lru_cache
    from typing import Optional, Dict, Any
    from urllib.parse import quote
    from SPARQLWrapper import SPARQLWrapper, JSON

    # ====================================================================
    # CONFIGURATION
    # ====================================================================

    CONFIG = {
        "cdk_base": "https://www.simolecule.com/cdkdepict/depict/cot/svg",
        "color_hyperlink": "#006699",
        "max_retries": 3,
        "page_size_default": 15,
        "page_size_export": 25,
        "query_limit": 1_000_000,
        "retry_backoff": 2,
        "user_agent": "LOTUS Explorer/0.0.1",
    }

    # Shared SPARQL instance
    SPARQL = SPARQLWrapper("https://query.wikidata.org/sparql")
    SPARQL.setReturnFormat(JSON)
    SPARQL.addCustomHttpHeader("User-Agent", CONFIG["user_agent"])


@app.function
def build_taxon_search_query(taxon_name: str) -> str:
    return f"""
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


@app.function
def build_compounds_query(qid: str) -> str:
    return f"""
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
    LIMIT {CONFIG["query_limit"]}
    """


@app.function
def execute_sparql(
    query: str, max_retries: int = CONFIG["max_retries"]
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            SPARQL.setQuery(query)
            return SPARQL.query().convert()
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Query failed after {max_retries} attempts: {str(e)}")
            time.sleep(CONFIG["retry_backoff"] * (2**attempt))


@app.function
@lru_cache(maxsize=128)
def extract_qid(url: str) -> str:
    return url.replace("http://www.wikidata.org/entity/", "")


@app.function
@lru_cache(maxsize=1024)
def create_structure_image_url(smiles: str) -> str:
    if not smiles:
        return "https://via.placeholder.com/120x120?text=No+SMILES"
    encoded_smiles = quote(smiles)
    return f"{CONFIG['cdk_base']}?smi={encoded_smiles}&annotate=cip"


@app.function
def taxon_name_to_qid(taxon_name: str) -> Optional[str]:
    try:
        query = build_taxon_search_query(taxon_name)
        results = execute_sparql(query)
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            return extract_qid(bindings[0].get("item", {}).get("value", ""))
    except Exception:
        pass
    return None


@app.function
def get_binding_value(binding: Dict[str, Any], key: str, default: str = "") -> str:
    return binding.get(key, {}).get("value", default)


@app.function
def create_link(url: str, text: str) -> mo.Html:
    color = CONFIG["color_hyperlink"]
    safe_text = text or url or ""
    safe_url = url or "#"
    return mo.Html(
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
        f'style="color: {color}; text-decoration: none; '
        f'border-bottom: 1px solid transparent; transition: border-color 0.2s;" '
        f"onmouseover=\"this.style.borderColor='{color}'\" "
        f"onmouseout=\"this.style.borderColor='transparent'\">{safe_text}</a>"
    )


@app.function
def apply_year_filter(
    df: pl.DataFrame, year_start: Optional[int], year_end: Optional[int]
) -> pl.DataFrame:
    if year_start is None or year_end is None or "pub_date" not in df.columns:
        return df
    return df.filter(
        pl.col("pub_date").is_null()
        | (
            (pl.col("pub_date").dt.year() >= year_start)
            & (pl.col("pub_date").dt.year() <= year_end)
        )
    )


@app.function
def query_wikidata(
    qid: str, year_start: Optional[int] = None, year_end: Optional[int] = None
) -> pl.DataFrame:
    query = build_compounds_query(qid)
    results = execute_sparql(query)
    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return pl.DataFrame()

    rows = []
    for b in bindings:
        pub_date_raw = get_binding_value(b, "pub_date", None)
        doi = get_binding_value(b, "ref_doi")
        if doi and doi.startswith("http"):
            doi = doi.split("doi.org/")[-1]

        rows.append(
            {
                "structure": get_binding_value(b, "structure"),
                "name": get_binding_value(b, "structureLabel"),
                "inchikey": get_binding_value(b, "inchikey"),
                "smiles": get_binding_value(b, "smiles_iso")
                or get_binding_value(b, "smiles_conn"),
                "taxon_name": get_binding_value(b, "taxon_name"),
                "taxon": get_binding_value(b, "taxon"),
                "ref_title": get_binding_value(b, "ref_title"),
                "ref_doi": doi,
                "reference": get_binding_value(b, "ref_qid"),
                "pub_date": pub_date_raw,
            }
        )

    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.when(pl.col("pub_date").is_not_null())
        .then(
            pl.col("pub_date").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False
            )
        )
        .otherwise(None)
        .alias("pub_date")
    )
    df = df.with_columns(pl.col("pub_date").dt.date())
    df = apply_year_filter(df, year_start, year_end)
    return df.unique(subset=["structure", "taxon", "reference"], keep="first").sort(
        "name"
    )


@app.function
def create_display_row(row: Dict[str, str]) -> Dict[str, Any]:
    img_url = create_structure_image_url(row["smiles"])
    struct_qid = extract_qid(row["structure"])
    taxon_qid = extract_qid(row["taxon"])
    ref_qid = extract_qid(row["reference"])
    ref_title = row["ref_title"] or "—"
    doi = row["ref_doi"]

    # Build hyperlinks
    struct_link = (
        create_link(f"https://www.wikidata.org/wiki/{struct_qid}", struct_qid)
        if struct_qid
        else "—"
    )
    taxon_link = (
        create_link(f"https://www.wikidata.org/wiki/{taxon_qid}", taxon_qid)
        if taxon_qid
        else "—"
    )
    ref_link = (
        create_link(f"https://www.wikidata.org/wiki/{ref_qid}", ref_qid)
        if ref_qid
        else "—"
    )
    doi_link = create_link(f"https://doi.org/{doi}", doi) if doi else "—"

    return {
        "2D Depiction": mo.image(src=img_url),
        "Compound": row["name"],
        "Compound SMILES": row["smiles"],
        "Compound InChIKey": row["inchikey"],
        "Taxon": row["taxon_name"],
        "Reference title": ref_title,
        "Reference DOI": doi_link,
        "Compound QID": struct_link,
        "Taxon QID": taxon_link,
        "Reference QID": ref_link,
    }


@app.function
def prepare_export_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
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
            pl.col("taxon_name"),
            pl.col("ref_title").alias("reference_title"),
            pl.col("ref_doi").alias("reference_doi"),
            pl.col("pub_date").alias("reference_date"),
            "compound_qid",
            "taxon_qid",
            "reference_qid",
        ]
    )


@app.cell
def _():
    mo.md("""
    # 🌿 LOTUS Wikidata Explorer

    Explore chemical compounds from [LOTUS](https://doi.org/10.7554/eLife.70780) and 
    [Wikidata](https://www.wikidata.org/) for any taxon.

    Enter a taxon name to discover chemical compounds found in organisms of that taxonomic group.
    """)
    return


@app.cell
def _():
    current_year = datetime.now().year

    taxon_input = mo.ui.text(
        value="Gentiana lutea",
        label="🔬 Taxon name",
        placeholder="e.g., Swertia, Artemisia, Homo sapiens, ...",
        full_width=True,
    )

    year_filter = mo.ui.checkbox(label="📅 Filter by publication year", value=False)

    year_start = mo.ui.number(
        value=1900, start=1700, stop=current_year, label="Start year", full_width=True
    )

    year_end = mo.ui.number(
        value=current_year,
        start=1700,
        stop=current_year,
        label="End year",
        full_width=True,
    )

    run_button = mo.ui.run_button(label="🔍 Search Wikidata")
    return run_button, taxon_input, year_end, year_filter, year_start


@app.cell
def _(run_button, taxon_input, year_end, year_filter, year_start):
    mo.vstack(
        [
            mo.md("## Search Parameters"),
            taxon_input,
            mo.hstack([year_filter], justify="start"),
            mo.hstack([year_start, year_end], gap=2, widths="equal")
            if year_filter.value
            else mo.Html(""),
            run_button,
        ]
    )
    return


@app.cell
def _(run_button, taxon_input, year_end, year_filter, year_start):
    if not run_button.value:
        results_df = None
        qid = None
    else:
        taxon_name = taxon_input.value.strip()
        start_time = time.time()
        with mo.status.spinner(title=f"🔎 Querying Wikidata for {taxon_name}..."):
            qid = taxon_name_to_qid(taxon_name)
            if not qid:
                mo.stop(
                    True,
                    mo.callout(
                        mo.md(
                            f"**Taxon not found:** Could not find '{taxon_name}' in Wikidata. Please check the spelling or try a different taxonomic name."
                        ),
                        kind="warn",
                    ),
                )

            try:
                y_start = year_start.value if year_filter.value else None
                y_end = year_end.value if year_filter.value else None
                results_df = query_wikidata(qid, y_start, y_end)
            except Exception as e:
                mo.stop(
                    True, mo.callout(mo.md(f"**Query Error:** {str(e)}"), kind="danger")
                )
        elapsed = round(time.time() - start_time, 2)
        mo.md(f"⏱️ Query completed in **{elapsed}s**.")
    return qid, results_df


@app.cell
def _(qid, results_df, run_button, taxon_input):
    if not run_button.value or results_df is None:
        summary_display = mo.Html("")
    elif len(results_df) == 0:
        summary_display = mo.callout(
            mo.md(
                f"No natural products found for **{taxon_input.value}** ({create_link(f'https://www.wikidata.org/wiki/{qid}', qid)}) with the current filters."
            ),
            kind="warn",
        )
    else:
        n_compounds = results_df.n_unique(subset=["structure"])
        n_taxa = results_df.n_unique(subset=["taxon"])
        n_refs = results_df.n_unique(subset=["reference"])

        # Choose singular/plural dynamically
        compound_label = "🧪 Compound" if n_compounds == 1 else "🧪 Compounds"
        taxon_label = "🌱 Taxon" if n_taxa == 1 else "🌱 Taxa"
        reference_label = "📚 Reference" if n_refs == 1 else "📚 References"
        entry_label = "📝 Entry" if len(results_df) == 1 else "📝 Entries"

        summary_display = mo.vstack(
            [
                mo.md(
                    f"""
            ## Results Summary

            Found data for **{taxon_input.value}** {create_link(f"https://www.wikidata.org/wiki/{qid}", qid)}
            """
                ),
                mo.hstack(
                    [
                        mo.stat(
                            value=str(n_compounds), label=compound_label, bordered=True
                        ),
                        mo.stat(value=str(n_taxa), label=taxon_label, bordered=True),
                        mo.stat(
                            value=str(n_refs), label=reference_label, bordered=True
                        ),
                        mo.stat(
                            value=str(len(results_df)), label=entry_label, bordered=True
                        ),
                    ],
                    gap=2,
                    justify="start",
                    wrap=True,
                ),
            ]
        )

    summary_display
    return


@app.cell
def _(results_df, run_button):
    if not run_button.value or results_df is None:
        table_output = None
    elif len(results_df) == 0:
        table_output = mo.callout(
            mo.md("No compounds match your search criteria."), kind="neutral"
        )
    else:
        # Create display table
        display_data = [
            create_display_row(row) for row in results_df.iter_rows(named=True)
        ]

        display_table = mo.ui.table(
            display_data,
            selection=None,
            page_size=CONFIG["page_size_default"],
            show_column_summaries=False,
        )

        # Create export table (preserves pub_date as actual date)
        export_df = prepare_export_dataframe(results_df)
        export_table = mo.ui.table(
            export_df,
            selection=None,
            page_size=CONFIG["page_size_export"],
            show_column_summaries=False,
        )

        table_output = mo.vstack(
            [
                mo.md("### Data Tables"),
                mo.ui.tabs({"🖼️  Display": display_table, "📥 Export": export_table}),
            ]
        )

    table_output
    return


@app.cell
def _():
    mo.md(
        """
    ---
    **Data:** <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> & <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a>  |  
    **Code:** <a href="https://github.com/cdk/depict" style="color:#339966;">CDK Depict</a> & 
    <a href="https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py" style="color:#339966;">lotus_wikidata_explorer.py</a>  |  
    **License:** <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#006699;">CC0 1.0</a> for data & <a href="https://www.gnu.org/licenses/gpl-3.0.html" style="color:#006699;">GPL-3.0</a> for code
    """
    )
    return


if __name__ == "__main__":
    app.run()
