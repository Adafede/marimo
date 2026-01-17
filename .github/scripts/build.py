"""
Build script for marimo notebooks.

This script exports marimo notebooks to HTML/WebAssembly format and generates
an index.html file that lists all the notebooks. It handles both regular notebooks
(from the notebooks/ directory) and apps (from the apps/ directory).

For apps, it automatically inlines modules from modules/ before exporting.

The script can be run from the command line with optional arguments:
    uv run .github/scripts/build.py [--output-dir OUTPUT_DIR]

The exported files will be placed in the specified output directory (default: _site).
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.3",
#     "fire==0.7.0",
#     "loguru==0.7.0"
# ]
# ///

import subprocess
import tempfile
import re
from typing import List, Union, Set
from pathlib import Path

import jinja2
import fire

from loguru import logger


def find_imported_modules(notebook_path: Path) -> Set[str]:
    """Find all modules imported from 'modules.*' in the notebook."""
    with open(notebook_path, "r") as f:
        content = f.read()

    imports = set()

    # Pattern 1: from modules.x.y import z
    pattern1 = r"from\s+(modules\.[\w.]+)\s+import"
    imports.update(re.findall(pattern1, content))

    # Pattern 2: import modules.x.y
    pattern2 = r"import\s+(modules\.[\w.]+)"
    imports.update(re.findall(pattern2, content))

    return imports


def find_module_dependencies(module_path: Path, module_name: str) -> Set[str]:
    """Recursively find all dependencies of a module."""
    if not module_path.exists():
        return set()

    with open(module_path, "r") as f:
        content = f.read()

    dependencies = set()

    # Find absolute imports from modules.*
    pattern1 = r"from\s+(modules\.[\w.]+)\s+import"
    dependencies.update(re.findall(pattern1, content))

    pattern2 = r"import\s+(modules\.[\w.]+)"
    dependencies.update(re.findall(pattern2, content))

    # Find relative imports and convert to absolute
    pattern3 = r"from\s+\.([\w.]+)\s+import"
    relative_imports = re.findall(pattern3, content)

    parent_module = ".".join(module_name.split(".")[:-1])
    for rel_import in relative_imports:
        absolute_import = f"{parent_module}.{rel_import}"
        dependencies.add(absolute_import)

    return dependencies


def get_all_required_modules(notebook_path: Path, public_path: Path) -> Set[str]:
    """Get all modules required by the notebook, including transitive dependencies."""
    direct_imports = find_imported_modules(notebook_path)

    all_modules = set()
    to_process = list(direct_imports)
    processed = set()

    while to_process:
        module_name = to_process.pop()
        if module_name in processed:
            continue

        processed.add(module_name)
        all_modules.add(module_name)

        # Find this module's dependencies
        module_file = public_path / f"{module_name.replace('.', '/')}.py"
        deps = find_module_dependencies(module_file, module_name)

        for dep in deps:
            if dep not in processed:
                to_process.append(dep)

    return all_modules


def convert_relative_to_absolute_imports(code: str, module_name: str) -> str:
    """Convert relative imports to absolute imports in module code."""
    parent_module = ".".join(module_name.split(".")[:-1])

    # Convert "from .x import y" to "from modules.parent.x import y"
    def replace_relative(match):
        relative_part = match.group(1)
        return f"from {parent_module}.{relative_part} import"

    code = re.sub(r"from\s+\.([\w.]+)\s+import", replace_relative, code)

    return code


def inline_modules(notebook_path: Path, output_path: Path, public_path: Path):
    """Inline only required modules into the notebook."""

    # Get all required modules
    required_modules = get_all_required_modules(notebook_path, public_path)

    if not required_modules:
        logger.info(f"No modules to inline for {notebook_path.name}")
        # Just copy the original
        with open(notebook_path, "r") as f:
            notebook_code = f.read()
        with open(output_path, "w") as f:
            f.write(notebook_code)
        return

    logger.info(
        f"Inlining {len(required_modules)} modules for {notebook_path.name}: {sorted(required_modules)}"
    )

    # Sort modules by dependency order (modules without deps first)
    module_graph = {}
    for module_name in required_modules:
        module_file = public_path / f"{module_name.replace('.', '/')}.py"
        deps = find_module_dependencies(module_file, module_name)
        # Only keep deps that are in our required set
        deps = deps & required_modules
        module_graph[module_name] = deps

    # Topological sort
    sorted_modules = []
    remaining = set(required_modules)

    while remaining:
        # Find modules with no remaining dependencies
        no_deps = [m for m in remaining if not (module_graph[m] & remaining)]
        if not no_deps:
            # Circular dependency - just add remaining in arbitrary order
            no_deps = list(remaining)

        sorted_modules.extend(sorted(no_deps))
        remaining -= set(no_deps)

    # Read and inline modules in dependency order
    # Collect all module code
    module_code_parts = []

    for module_name in sorted_modules:
        module_file = public_path / f"{module_name.replace('.', '/')}.py"

        if not module_file.exists():
            logger.warning(f"Module file not found: {module_file}")
            continue

        with open(module_file, "r") as f:
            module_code = f.read()

        # Convert relative imports to absolute
        module_code = convert_relative_to_absolute_imports(module_code, module_name)

        module_code_parts.append(f"    # --- {module_name} ---")
        # Indent all lines of module code by 4 spaces
        indented_code = "\n".join("    " + line if line.strip() else "" for line in module_code.split("\n"))
        module_code_parts.append(indented_code)
        module_code_parts.append("")

    # Build the inlined code as an app.setup block
    inlined_cell = []
    inlined_cell.append("")
    inlined_cell.append("")
    inlined_cell.append("with app.setup:")
    inlined_cell.append("    # === AUTO-INLINED MODULES ===")
    inlined_cell.append("    # This code was automatically inlined from modules/")
    inlined_cell.append("    import marimo as mo")
    inlined_cell.append("")
    inlined_cell.extend(module_code_parts)


    inlined_cell.append("")

    inlined_code_str = "\n".join(inlined_cell)

    # Read original notebook
    with open(notebook_path, "r") as f:
        notebook_code = f.read()

    # Remove the setup_local cell if it exists (used only for local dev)
    notebook_code = re.sub(
        r"@app\.cell\s+def\s+setup_local\(\):.*?return\s*\n",
        "",
        notebook_code,
        flags=re.DOTALL,
    )

    # Remove all 'from modules.*' and 'import modules.*' lines since they are now inlined
    notebook_code = re.sub(r"^\s*from\s+modules\.[\w.]+\s+import\s+[^\n]+\n", "", notebook_code, flags=re.MULTILINE)
    notebook_code = re.sub(r"^\s*import\s+modules\.[\w.]+\s*\n", "", notebook_code, flags=re.MULTILINE)

    # Insert the inlined modules cell right after the app = marimo.App(...) line
    # Find the app setup line pattern
    app_setup_pattern = r"(app\s*=\s*marimo\.App\([^)]*\))"
    match = re.search(app_setup_pattern, notebook_code)

    if match:
        # Insert inlined code after the app setup line
        insert_pos = match.end()
        final_code = notebook_code[:insert_pos] + inlined_code_str + notebook_code[insert_pos:]
    else:
        # Fallback: just prepend (shouldn't happen for valid marimo notebooks)
        logger.warning("Could not find 'app = marimo.App()' line, prepending inlined code")
        final_code = inlined_code_str + "\n\n" + notebook_code

    # Write to output
    with open(output_path, "w") as f:
        f.write(final_code)

    logger.info(f"Successfully inlined modules into {output_path}")


def _export_html_wasm(
    notebook_path: Path, output_dir: Path, as_app: bool = False
) -> bool:
    """Export a single marimo notebook to HTML/WebAssembly format.

    This function takes a marimo notebook (.py file) and exports it to HTML/WebAssembly format.
    If as_app is True, the notebook is exported in "run" mode with code hidden, suitable for
    applications. Otherwise, it's exported in "edit" mode, suitable for interactive notebooks.

    For apps, if a modules/ directory exists, modules will be automatically
    inlined before export.

    Args:
        notebook_path (Path): Path to the marimo notebook (.py file) to export
        output_dir (Path): Directory where the exported HTML file will be saved
        as_app (bool, optional): Whether to export as an app (run mode) or notebook (edit mode).
                                Defaults to False.

    Returns:
        bool: True if export succeeded, False otherwise
    """
    temp_path = None

    # If exporting as app and modules/ exists, inline modules first
    if as_app:
        public_dir = notebook_path.parent / ".."
        if public_dir.exists() and public_dir.is_dir():
            # Create temporary inlined version
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                temp_path = Path(tmp.name)

            logger.info(f"Inlining modules from {public_dir} for {notebook_path.name}")
            inline_modules(notebook_path, temp_path, public_dir)

            # Export the inlined version instead
            notebook_to_export = temp_path
        else:
            notebook_to_export = notebook_path
    else:
        notebook_to_export = notebook_path

    # Convert .py extension to .html for the output file
    output_path: Path = notebook_path.with_suffix(".html")

    # Base command for marimo export
    cmd: List[str] = ["uvx", "marimo", "export", "html-wasm", "--sandbox"]

    # Configure export mode based on whether it's an app or a notebook
    if as_app:
        logger.info(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(
            ["--mode", "run", "--no-show-code"]
        )  # Apps run in "run" mode with hidden code
    else:
        logger.info(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])  # Notebooks run in "edit" mode

    try:
        # Create full output path and ensure directory exists
        output_file: Path = output_dir / notebook_path.with_suffix(".html")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Add notebook path and output file to command
        cmd.extend([str(notebook_to_export), "-o", str(output_file)])

        # Run marimo export command
        logger.debug(f"Running command: {cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully exported {notebook_path}")

        return True
    except subprocess.CalledProcessError as e:
        # Handle marimo export errors
        logger.error(f"Error exporting {notebook_path}:")
        logger.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error exporting {notebook_path}: {e}")
        return False
    finally:
        # Clean up temp file if we created one
        if temp_path and temp_path.exists():
            # temp_path.unlink()
            pass


def _generate_index(
    output_dir: Path,
    template_file: Path,
    notebooks_data: List[dict] | None = None,
    apps_data: List[dict] | None = None,
) -> None:
    """Generate an index.html file that lists all the notebooks.

    This function creates an HTML index page that displays links to all the exported
    notebooks. The index page includes the marimo logo and displays each notebook
    with a formatted title and a link to open it.

    Args:
        notebooks_data (List[dict]): List of dictionaries with data for notebooks
        apps_data (List[dict]): List of dictionaries with data for apps
        output_dir (Path): Directory where the index.html file will be saved
        template_file (Path, optional): Path to the template file. If None, uses the default template.

    Returns:
        None
    """
    logger.info("Generating index.html")

    # Create the full path for the index.html file
    index_path: Path = output_dir / "index.html"

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Set up Jinja2 environment and load template
        template_dir = template_file.parent
        template_name = template_file.name
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template_name)

        # Render the template with notebook and app data
        rendered_html = template.render(notebooks=notebooks_data, apps=apps_data)

        # Write the rendered HTML to the index.html file
        with open(index_path, "w") as f:
            f.write(rendered_html)
        logger.info(f"Successfully generated index.html at {index_path}")

    except IOError as e:
        # Handle file I/O errors
        logger.error(f"Error generating index.html: {e}")
    except jinja2.exceptions.TemplateError as e:
        # Handle template errors
        logger.error(f"Error rendering template: {e}")


def _export(folder: Path, output_dir: Path, as_app: bool = False) -> List[dict]:
    """Export all marimo notebooks in a folder to HTML/WebAssembly format.

    This function finds all Python files in the specified folder and exports them
    to HTML/WebAssembly format using the export_html_wasm function. It returns a
    list of dictionaries containing the data needed for the template.

    Args:
        folder (Path): Path to the folder containing marimo notebooks
        output_dir (Path): Directory where the exported HTML files will be saved
        as_app (bool, optional): Whether to export as apps (run mode) or notebooks (edit mode).

    Returns:
        List[dict]: List of dictionaries with "display_name" and "html_path" for each notebook
    """
    # Check if the folder exists
    if not folder.exists():
        logger.warning(f"Directory not found: {folder}")
        return []

    # Find all Python files recursively in the folder, excluding public/ directories
    all_notebooks = list(folder.rglob("*.py"))

    # Filter out files in public/ directories
    notebooks = [nb for nb in all_notebooks if "public" not in nb.parts]

    if len(all_notebooks) != len(notebooks):
        logger.debug(
            f"Filtered out {len(all_notebooks) - len(notebooks)} files from public/ directories"
        )

    logger.debug(f"Found {len(notebooks)} Python files in {folder}")

    # Exit if no notebooks were found
    if not notebooks:
        logger.warning(f"No notebooks found in {folder}!")
        return []

    # For each successfully exported notebook, add its data to the notebook_data list
    notebook_data = [
        {
            "display_name": (nb.stem.replace("_", " ").title()),
            "html_path": str(nb.with_suffix(".html")),
        }
        for nb in notebooks
        if _export_html_wasm(nb, output_dir, as_app=as_app)
    ]

    logger.info(
        f"Successfully exported {len(notebook_data)} out of {len(notebooks)} files from {folder}"
    )
    return notebook_data


def main(
    output_dir: Union[str, Path] = "_site",
    template: Union[str, Path] = "templates/tailwind.html.j2",
) -> None:
    """Main function to export marimo notebooks.

    This function:
    1. Parses command line arguments
    2. Exports all marimo notebooks in the 'notebooks' and 'apps' directories
    3. For apps, automatically inlines modules from modules/ directories
    4. Generates an index.html file that lists all the notebooks

    Command line arguments:
        --output-dir: Directory where the exported files will be saved (default: _site)
        --template: Path to the template file (default: templates/index.html.j2)

    Returns:
        None
    """
    logger.info("Starting marimo build process")

    # Convert output_dir explicitly to Path (not done by fire)
    output_dir: Path = Path(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert template to Path if provided
    template_file: Path = Path(template)
    logger.info(f"Using template file: {template_file}")

    # Export notebooks from the notebooks/ directory
    notebooks_data = _export(Path("notebooks"), output_dir, as_app=False)

    # Export apps from the apps/ directory
    apps_data = _export(Path("apps"), output_dir, as_app=True)

    # Exit if no notebooks or apps were found
    if not notebooks_data and not apps_data:
        logger.warning("No notebooks or apps found!")
        return

    # Generate the index.html file that lists all notebooks and apps
    _generate_index(
        output_dir=output_dir,
        notebooks_data=notebooks_data,
        apps_data=apps_data,
        template_file=template_file,
    )

    logger.info(f"Build completed successfully. Output directory: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
