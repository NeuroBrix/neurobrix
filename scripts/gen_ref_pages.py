"""Generate API reference pages from source code.

This script is run by mkdocs-gen-files during `mkdocs build`.
It walks the neurobrix package and creates a markdown page for each module,
using mkdocstrings to pull docstrings directly from the code.

This means API docs are ALWAYS in sync with the code — no manual updates needed.
"""

from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path("src")
package = src / "neurobrix"

# Modules to document (public API only)
PUBLIC_MODULES = [
    "cli",
    "core/runtime",
    "core/prism",
    "core/dtype",
    "core/flow",
    "core/module",
    "nbx",
    "kernels",
    "serving",
]

# Files to skip (internal, not public API)
SKIP_FILES = {
    "__pycache__",
    "__init__.py",
    "__main__.py",
    "CLAUDE.md",
}

SKIP_DIRS = {
    "__pycache__",
    "config",  # YAML configs, not Python API
    "triton_kernels_ref",  # Vendored reference kernels, not public API
}

for module_path in PUBLIC_MODULES:
    full_path = package / module_path
    if not full_path.exists():
        continue

    if full_path.is_file():
        files = [full_path]
    else:
        files = sorted(full_path.rglob("*.py"))

    for path in files:
        if path.name in SKIP_FILES:
            continue
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue

        # Convert path to module dotted name
        module_parts = list(path.relative_to(src).with_suffix("").parts)
        doc_path = Path("reference/api", *module_parts).with_suffix(".md")

        # Build the identifier for mkdocstrings
        identifier = ".".join(module_parts)

        # Create the markdown page
        with mkdocs_gen_files.open(doc_path, "w") as fd:
            fd.write(f"# `{identifier}`\n\n")
            fd.write(f"::: {identifier}\n")

        # Add to navigation
        nav_parts = module_parts[1:]  # Skip "neurobrix" prefix for nav
        nav[tuple(nav_parts)] = str(doc_path)

        mkdocs_gen_files.set_edit_path(doc_path, f"../{path}")

# Write the SUMMARY.md for literate-nav
with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
