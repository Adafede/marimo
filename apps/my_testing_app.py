# /// script
# requires-python = "==3.13.*"
# dependencies = [
#    "adafedemarimo @ file:///Users/adrutz/Git/marimo",
#     "marimo",
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
    from adafedemarimo.foo import Bar


@app.cell
def bar():
    Bar()
    return


if __name__ == "__main__":
    app.run()
