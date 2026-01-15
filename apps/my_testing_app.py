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
    from pathlib import Path
    
    nb_dir = mo.notebook_dir()
    files = sorted(p.name for p in nb_dir.iterdir())
    mo.plain_text("\n".join(files))
    return


if __name__ == "__main__":
    app.run()
