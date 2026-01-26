# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
# ]
# ///
"""
Test app where parameters get erased by `marimo check`
"""

import marimo

__generated_with = "0.18.1"
app = marimo.App(
    width="medium",
    app_title="Test app",
    html_head_file="dokieli.html",
)

with app.setup:
    import marimo as mo


@app.cell
def hello():
    mo.md("""
    Hello World
    """)
    return


if __name__ == "__main__":
    app.run()
