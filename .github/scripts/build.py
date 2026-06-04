"""Build script for marimo notebooks.

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
import re
import shutil
from typing import List, Union, Set
from pathlib import Path

import jinja2
import fire

from loguru import logger


def copy_public_directories(source_dir: Path, output_dir: Path) -> None:
    """Copy all public/ directories from source to output directory.

    The copy preserves each directory's relative path below ``source_dir`` and
    replaces any pre-existing destination directory.

    Parameters
    ----------
    source_dir : Path
        Root directory scanned recursively for ``public`` directories.
    output_dir : Path
        Root output directory where matching ``public`` paths are written.

    """
    if not source_dir.exists():
        logger.warning(f"Source directory not found: {source_dir}")
        return

    # Find all public directories recursively
    public_dirs = list(source_dir.rglob("public"))

    if not public_dirs:
        logger.debug(f"No public directories found in {source_dir}")
        return

    logger.info(
        f"Found {len(public_dirs)} public directory/directories in {source_dir}",
    )

    for public_dir in public_dirs:
        # Calculate the relative path from source_dir to this public dir
        relative_path = public_dir.relative_to(source_dir)

        # Create the corresponding path in output_dir
        dest_path = output_dir / relative_path

        try:
            # Remove existing directory if it exists
            if dest_path.exists():
                shutil.rmtree(dest_path)

            # Copy the entire public directory
            shutil.copytree(public_dir, dest_path)
            logger.info(f"Copied {public_dir} -> {dest_path}")

        except Exception as e:
            logger.error(f"Error copying {public_dir} to {dest_path}: {e}")


def find_imported_modules(notebook_path: Path) -> Set[str]:
    """Find ``modules.*`` imports declared in a notebook script.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook Python file.

    Returns
    -------
    Set[str]
        Imported module paths such as ``modules.text.strings``.

    """
    with open(notebook_path) as f:
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
    """Find direct module dependencies referenced by a module source file.

    Parameters
    ----------
    module_path : Path
        Filesystem path to the module source.
    module_name : str
        Dotted module path used to resolve relative imports.

    Returns
    -------
    Set[str]
        Absolute dotted module paths found in import statements.

    """
    if not module_path.exists():
        return set()

    with open(module_path) as f:
        content = f.read()

    dependencies = set()

    # Find absolute imports from modules.*
    pattern1 = r"from\s+(modules\.[\w.]+)\s+import"
    dependencies.update(re.findall(pattern1, content))

    pattern2 = r"import\s+(modules\.[\w.]+)"
    dependencies.update(re.findall(pattern2, content))

    # Find relative imports and convert to absolute
    # Match patterns like: from .foo, from ..foo, from ...foo
    pattern3 = r"from\s+(\.+)([\w.]*)\s+import"
    relative_imports = re.findall(pattern3, content)

    module_parts = module_name.split(".")
    for dots, rel_import in relative_imports:
        # Number of dots determines how many levels to go up
        # 1 dot = same package (go up 1 level from current module)
        # 2 dots = parent package (go up 2 levels)
        # etc.
        levels_up = len(dots)
        if levels_up > len(module_parts) - 1:
            continue  # Can't go up that many levels

        # Get the base package by going up the appropriate number of levels
        base_parts = module_parts[:-levels_up] if levels_up > 0 else module_parts[:-1]

        if rel_import:
            absolute_import = ".".join(base_parts + [rel_import])
        else:
            # from . import foo - imports from __init__.py
            absolute_import = ".".join(base_parts)

        dependencies.add(absolute_import)

    return dependencies


def get_all_required_modules(notebook_path: Path, public_path: Path) -> Set[str]:
    """Resolve the transitive closure of notebook module dependencies.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook Python file.
    public_path : Path
        Repository root used to locate module source files.

    Returns
    -------
    Set[str]
        All direct and transitive ``modules.*`` dependencies.

    """
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
    """Convert relative import statements in module code to absolute imports.

    Parameters
    ----------
    code : str
        Module source code.
    module_name : str
        Dotted module path for the source code being rewritten.

    Returns
    -------
    str
        Rewritten source code with relative imports normalized.

    """
    module_parts = module_name.split(".")

    # Convert "from .x import y", "from ..x import y", etc. to absolute imports
    def replace_relative(match):
        dots = match.group(1)
        relative_part = match.group(2)

        # Number of dots determines how many levels to go up
        levels_up = len(dots)
        if levels_up > len(module_parts) - 1:
            return match.group(0)  # Can't convert, keep original

        # Get the base package by going up the appropriate number of levels
        base_parts = module_parts[:-levels_up] if levels_up > 0 else module_parts[:-1]

        if relative_part:
            absolute_path = ".".join(base_parts + [relative_part])
        else:
            absolute_path = ".".join(base_parts)

        return f"from {absolute_path} import"

    code = re.sub(r"from\s+(\.+)([\w.]*)\s+import", replace_relative, code)

    return code


def inline_modules(notebook_path: Path, output_path: Path, public_path: Path):
    """Inline required local modules directly into a notebook script.

    Parameters
    ----------
    notebook_path : Path
        Path to the source notebook.
    output_path : Path
        Path where the transformed notebook is written.
    public_path : Path
        Repository root used to resolve ``modules.*`` imports.

    """
    # Get all required modules
    required_modules = get_all_required_modules(notebook_path, public_path)

    if not required_modules:
        logger.info(f"No modules to inline for {notebook_path.name}")
        # Just copy the original
        with open(notebook_path) as f:
            notebook_code = f.read()
        with open(output_path, "w") as f:
            f.write(notebook_code)
        return

    logger.info(
        f"Inlining {len(required_modules)} modules for {notebook_path.name}: {sorted(required_modules)}",
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
    # Build a map of module_name -> module_code for in-place replacement
    module_code_map = {}

    for module_name in sorted_modules:
        module_file = public_path / f"{module_name.replace('.', '/')}.py"

        if not module_file.exists():
            logger.warning(f"Module file not found: {module_file}")
            continue

        with open(module_file) as f:
            module_code = f.read()

        # Convert relative imports to absolute
        module_code = convert_relative_to_absolute_imports(module_code, module_name)

        # Store with the full module path as key
        module_code_map[module_name] = module_code

    # Track which modules have been inlined to avoid duplicates
    inlined_modules = set()

    # Read original notebook
    with open(notebook_path) as f:
        notebook_code = f.read()

    # Remove the setup_local cell if it exists (used only for local dev)
    notebook_code = re.sub(
        r"@app\.cell\s+def\s+setup_local\(\):.*?return\s*\n",
        "",
        notebook_code,
        flags=re.DOTALL,
    )

    # Function to get inlined code for a module and its dependencies
    def get_inlined_code_with_deps(
        module_path: str,
        indent: str,
        aliases: dict | None = None,
    ) -> str:
        """Return inlined code for a module and its transitive dependencies.

        Parameters
        ----------
        module_path : str
            Dotted module path.
        indent : str
            Indentation prefix used at the import site.
        aliases : dict | None
            Optional mapping of imported symbol names to aliases.

        Returns
        -------
        str
            Inlined code block, including alias assignments when applicable.

        """
        result_parts = []

        if module_path in inlined_modules:
            # Module already inlined - just add alias assignments at this indentation
            if aliases:
                alias_lines = []
                for original, alias in aliases.items():
                    if original != alias:
                        alias_lines.append(f"{indent}{alias} = {original}")
                if alias_lines:
                    return "\n".join(alias_lines) + "\n"
            return ""

        if module_path not in module_code_map:
            return ""  # Module not found

        code = module_code_map[module_path]

        # Find and inline dependencies FIRST (from the module's imports)
        # Look for 'from modules.x.y import' patterns in this module's code
        dep_pattern = r"from\s+(modules\.[\w.]+)\s+import"
        deps_in_code = re.findall(dep_pattern, code)

        # Sort and deduplicate for deterministic order
        for dep in sorted(set(deps_in_code)):
            if dep not in inlined_modules and dep in module_code_map:
                # Recursively inline the dependency first (no aliases for transitive deps)
                dep_code = get_inlined_code_with_deps(dep, indent, None)
                if dep_code:
                    result_parts.append(dep_code)

        # Mark this module as inlined BEFORE processing its code
        # to prevent infinite recursion with circular deps
        inlined_modules.add(module_path)

        # Normalize line endings (convert \r\n to \n)
        code = code.replace("\r\n", "\n").replace("\r", "\n")

        # Remove 'from modules.*' imports from the code since deps are now inlined
        # Handle multi-line imports with parentheses (including 'as' aliases)
        code = re.sub(
            r"^\s*from\s+modules\.[\w.]+\s+import\s*\([^)]*\)\n",
            "",
            code,
            flags=re.MULTILINE | re.DOTALL,
        )
        # Handle single-line imports (must not start with parenthesis after import)
        code = re.sub(
            r"^\s*from\s+modules\.[\w.]+\s+import\s+(?!\()[^\n]+\n",
            "",
            code,
            flags=re.MULTILINE,
        )
        code = re.sub(
            r"^\s*import\s+modules\.[\w.]+\s*\n",
            "",
            code,
            flags=re.MULTILINE,
        )

        # Normalize: collapse multiple consecutive blank lines into one
        code = re.sub(r"\n\n+", "\n\n", code)

        # Remove leading/trailing whitespace but keep internal structure
        code = code.strip()

        # Indent and add this module's code
        indented_lines = []
        indented_lines.append(f"{indent}# --- inlined from {module_path} ---")

        # Split code into lines and process
        lines = code.split("\n")
        for line in lines:
            if line.strip():
                indented_lines.append(f"{indent}{line}")
            elif line == "":
                # Only add one blank line, skip if previous was also blank
                if indented_lines and indented_lines[-1] != "":
                    indented_lines.append("")

        # Add trailing blank line only if not already present
        if indented_lines and indented_lines[-1] != "":
            indented_lines.append("")

        result_parts.append("\n".join(indented_lines))

        # Add alias assignments AFTER the module code
        if aliases:
            alias_lines = []
            for original, alias in aliases.items():
                if original != alias:
                    alias_lines.append(f"{indent}{alias} = {original}")
            if alias_lines:
                result_parts.append("\n".join(alias_lines) + "\n")

        return "\n".join(result_parts)

    def parse_import_aliases(import_text: str) -> dict:
        """Parse import targets into ``name -> alias`` mappings.

        Examples
        --------
        ``"foo, bar"`` -> ``{"foo": "foo", "bar": "bar"}``

        ``"foo as f, bar as b"`` -> ``{"foo": "f", "bar": "b"}``

        ``"(foo as f, bar)"`` -> ``{"foo": "f", "bar": "bar"}``

        Parameters
        ----------
        import_text : str
            Raw text after the ``import`` keyword.

        Returns
        -------
        dict
            Mapping from original symbol names to in-scope aliases.

        """
        aliases = {}
        # Remove parentheses and normalize whitespace
        text = import_text.strip()
        if text.startswith("("):
            text = text[1:]
        if text.endswith(")"):
            text = text[:-1]

        # Split by comma and process each import
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue

            if " as " in item:
                original, alias = item.split(" as ", 1)
                aliases[original.strip()] = alias.strip()
            else:
                aliases[item] = item

        return aliases

    # Function to replace a single import with inlined code
    def replace_import_with_inline(match):
        full_match = match.group(0)
        indent = match.group(1)  # Capture the indentation
        module_path = match.group(2)  # e.g., "modules.text.strings.pluralize"

        # Extract the import names part (everything after 'import')
        import_match = re.search(r"\bimport\s+(.+)$", full_match, re.DOTALL)
        aliases = {}
        if import_match:
            import_text = import_match.group(1).rstrip("\n")
            aliases = parse_import_aliases(import_text)

        if module_path in module_code_map:
            return get_inlined_code_with_deps(module_path, indent, aliases)
        else:
            # Module not found in our map, keep the original import
            # (will be removed later if it's a modules.* import)
            return full_match

    # Process ALL imports in source order (both single-line and multi-line)
    # This is critical because the first import of a module must inline its code,
    # while subsequent imports only add aliases

    # Pattern for single-line imports: from modules.x.y import z
    single_line_pattern = r"^(\s*)from\s+(modules\.[\w.]+)\s+import\s+(?!\()[^\n]+\n"

    # Pattern for multi-line imports: from modules.x.y import (\n    z as alias,\n)
    multi_line_pattern = r"^(\s*)from\s+(modules\.[\w.]+)\s+import\s*\([^)]*\)\n"

    # Find all matches with their positions
    all_matches = []

    for match in re.finditer(single_line_pattern, notebook_code, flags=re.MULTILINE):
        all_matches.append((match.start(), match.end(), match))

    for match in re.finditer(
        multi_line_pattern,
        notebook_code,
        flags=re.MULTILINE | re.DOTALL,
    ):
        all_matches.append((match.start(), match.end(), match))

    # Sort by position (start index) to process in source order
    all_matches.sort(key=lambda x: x[0])

    # First pass: determine what each match should be replaced with (in source order)
    # This ensures the first import of a module gets the full code
    replacements = []
    for start, end, match in all_matches:
        replacement = replace_import_with_inline(match)
        replacements.append((start, end, replacement))

    # Second pass: apply replacements in reverse order (so positions don't shift)
    for start, end, replacement in reversed(replacements):
        notebook_code = notebook_code[:start] + replacement + notebook_code[end:]

    # Remove any remaining 'from modules.*' lines that weren't replaced
    # (e.g., modules we didn't have code for)
    notebook_code = re.sub(
        r"^\s*from\s+modules\.[\w.]+\s+import\s*\([^)]*\)\n",
        "",
        notebook_code,
        flags=re.MULTILINE | re.DOTALL,
    )
    notebook_code = re.sub(
        r"^\s*from\s+modules\.[\w.]+\s+import\s+(?!\()[^\n]+\n",
        "",
        notebook_code,
        flags=re.MULTILINE,
    )
    notebook_code = re.sub(
        r"^\s*import\s+modules\.[\w.]+\s*\n",
        "",
        notebook_code,
        flags=re.MULTILINE,
    )

    final_code = notebook_code

    # Write to output
    with open(output_path, "w") as f:
        f.write(final_code)

    logger.info(f"Successfully inlined modules into {output_path}")


def inject_simpleanalytics(html_path: Path) -> None:
    """Inject the Simple Analytics script tag before ``</body>``.

    Parameters
    ----------
    html_path : Path
        Path to the exported HTML file.

    """
    html = html_path.read_text()

    script = """
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
    """

    # Insert before </body>
    if "</body>" in html:
        html = html.replace("</body>", script + "\n</body>")
        html_path.write_text(html)


def _export_html_wasm(
    notebook_path: Path,
    output_dir: Path,
    as_app: bool = False,
) -> bool:
    """Export a single marimo notebook to HTML/WebAssembly format.

    When ``as_app`` is true, the notebook is exported in run mode and local
    modules are inlined before export if available.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook source file.
    output_dir : Path
        Output root directory.
    as_app : bool
        Whether to export in app/run mode. Default is ``False``.

    Returns
    -------
    bool
        ``True`` when export succeeds, otherwise ``False``.

    """
    inlined_path = None
    notebook_to_export = notebook_path

    # If exporting as app and modules/ exists, inline modules first
    if as_app:
        public_dir = notebook_path.parent / ".."
        if public_dir.exists() and public_dir.is_dir():
            # Create the inlined version directly in the output directory
            output_py_path = output_dir / notebook_path
            output_py_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Inlining modules from {public_dir} for {notebook_path.name}")
            inline_modules(notebook_path, output_py_path, public_dir)

            # Export the inlined version
            notebook_to_export = output_py_path
            inlined_path = output_py_path

    # Convert .py extension to .html for the output file
    output_path: Path = notebook_path.with_suffix(".html")

    # Base command for marimo export
    cmd: List[str] = ["uv", "run", "marimo", "export", "html-wasm", "--sandbox"]

    # Configure export mode based on whether it's an app or a notebook
    if as_app:
        logger.info(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(
            ["--mode", "run", "--no-show-code"],
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
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully exported {notebook_path}")
        inject_simpleanalytics(output_file)
        logger.info("Successfully added analytics")

        if inlined_path:
            logger.info(f"Inlined notebook saved to {inlined_path}")

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


def _generate_index(
    output_dir: Path,
    template_file: Path,
    notebooks_data: List[dict] | None = None,
    apps_data: List[dict] | None = None,
) -> None:
    """Generate the ``index.html`` page for exported notebooks and apps.

    Parameters
    ----------
    output_dir : Path
        Output root directory where ``index.html`` is written.
    template_file : Path
        Jinja template file used to render the index page.
    notebooks_data : List[dict] | None
        Notebook metadata for rendering notebook entries.
    apps_data : List[dict] | None
        App metadata for rendering app entries.

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

    except OSError as e:
        # Handle file I/O errors
        logger.error(f"Error generating index.html: {e}")
    except jinja2.exceptions.TemplateError as e:
        # Handle template errors
        logger.error(f"Error rendering template: {e}")


def _export(folder: Path, output_dir: Path, as_app: bool = False) -> List[dict]:
    """Export all marimo notebooks in a folder to HTML/WebAssembly format.

    Parameters
    ----------
    folder : Path
        Root folder scanned recursively for notebook scripts.
    output_dir : Path
        Output root directory.
    as_app : bool
        Whether to export files as apps (run mode). Default is ``False``.

    Returns
    -------
    List[dict]
        Metadata records for successfully exported notebooks.

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
            f"Filtered out {len(all_notebooks) - len(notebooks)} files from public/ directories",
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
        f"Successfully exported {len(notebook_data)} out of {len(notebooks)} files from {folder}",
    )
    return notebook_data


def main(
    output_dir: Union[str, Path] = "_site",
    template: Union[str, Path] = "templates/tailwind.html.j2",
) -> None:
    """Export marimo notebooks.

    The build process exports notebooks and apps, copies ``public`` assets, and
    renders the HTML index page.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Output directory. Default is ``"_site"``.
    template : Union[str, Path]
        Template path used for index generation.
        Default is ``"templates/tailwind.html.j2"``.

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

    # Copy public directories from both notebooks/ and apps/
    logger.info("Copying public directories...")
    copy_public_directories(Path("notebooks"), output_dir / Path("notebooks"))
    copy_public_directories(Path("apps"), output_dir / Path("apps"))

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
