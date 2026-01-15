# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "httpimport==1.4.1",
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
    import httpimport

    with httpimport.remote_repo(url="https://raw.githubusercontent.com/Adafede/marimo/main/modules"):
        # Now you can import the module
        from text import smiles


@app.cell
def _():
    smiles.validate_smiles("aaa")
    return

if __name__ == "__main__":
    app.run()
