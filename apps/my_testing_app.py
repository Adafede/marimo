# /// script
# requires-python = "==3.13.*"
# dependencies = [
#    "adafede-marimo @ git+https://github.com/adafede/marimo",
#    "marimo",
# ]
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(
    app_title="My testing app",
)

with app.setup:
    import marimo as mo
    from adafede_marimo.foo import Bar


@app.cell
def message_md():
    mo.md(Bar())
    return


if __name__ == "__main__":
    app.run()
