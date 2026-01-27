# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(app_title="Remote Demo")

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
    from modules.chem.cdk.depict.svg_from_smiles import svg_from_smiles
    return parse, svg_from_smiles


@app.cell
def example_1(parse):
    mo.show_code(parse("C₆H₁₂O₆"), position="above")
    return


@app.cell
def example_2(svg_from_smiles):
    mo.show_code(mo.image(svg_from_smiles("CCO")), position="above")
    return


if __name__ == "__main__":
    app.run()
