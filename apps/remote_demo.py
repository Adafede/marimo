# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "starlette==0.51.0",  # because of a nasty bug from 0.52.0
# ]
#
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(
    app_title="Remote Demo",
    css_file="public/custom.css",
    html_head_file="public/head.html",
)

with app.setup:
    import marimo as mo


@app.cell
def title():
    mo.md("""
    # Remote Import Demo
    """)
    return


@app.cell
def imports():
    # Toggle this flag for local vs remote development
    _USE_LOCAL = False  # Set to True for local development
    if _USE_LOCAL:
        import sys

        # Add your local module directory to the path
        # Adjust this path to where your "modules" folder is located locally
        sys.path.insert(0, ".")
    # Modules will be auto-inlined by the build script
    from modules.text.formula.parse import parse
    from modules.chem.cdk.depict.url_from_smiles import url_from_smiles
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
