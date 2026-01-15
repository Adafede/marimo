# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "adafedemarimo==0.1.0",
#     "marimo",
# ]
#
# [tool.uv.sources]
# adafedemarimo = { git = "https://github.com/adafede/marimo", rev = "main" }
#
# [tool.marimo.display]
# theme = "system"
# ///

"""
Testing app for adafede_marimo modules.
"""

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
