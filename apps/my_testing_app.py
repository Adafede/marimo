# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
# ]
#
#
# [tool.marimo.display]
# theme = "system"
# ///


import marimo

__generated_with = "0.19.2"
app = marimo.App(app_title="Package Demo")

with app.setup:
    import marimo as mo


@app.cell
def intro():
    mols_path = mo.notebook_location() / "mols.py"
    with open(mols_path, "r") as f:
    top_5_lines = f.readlines()[:5]

    mo.plain_text(top_5_lines)
    return


if __name__ == "__main__":
    app.run()
