# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "requests==2.32.5",
# ]
#
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(app_title="Remote Import Demo")

with app.setup:
    import marimo as mo


@app.cell
def title():
    mo.md("""
    # Remote Import Demo
    """)
    return


@app.cell
def _():
    # === COPY INTO YOUR APP SETUP ===
    import sys

    # === Import modules from /public ===
    sys.path.insert(0, str(mo.notebook_location() / "public"))

    mo.show_code()
    return


@app.cell
def imports():
    from modules.text.formula.parse import parse
    from modules.chem.cdk.depict.url_from_smiles import url_from_smiles

    mo.show_code()
    return parse, url_from_smiles


@app.cell
def example_1(parse):
    mo.show_code(parse("C₆H₁₂O₆"), position="above")
    return


@app.cell
def example_2(url_from_smiles):
    mo.show_code(mo.image(url_from_smiles("CCO")), position="above")
    return


if __name__ == "__main__":
    app.run()
