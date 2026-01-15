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
    with open(mo.notebook_dir() / ".." / "pyproject.toml") as f:
        contents = f.read()

    mo.plain_text(contents)
    return


if __name__ == "__main__":
    app.run()
