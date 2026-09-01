"""Convert .py scripts to well-structured .ipynb notebooks.

Keeps ALL comments as code cells (not markdown). Only the module-level
docstring at the very top becomes a markdown cell. Function/class bodies
(including nested functions) stay intact as single cells.

Regeneration guard
------------------
The .py files are the canonical source, but notebooks may also hold manual
edits (e.g. re-baselining a comparison). To avoid silently destroying that
manual work on every regeneration:

  * A sidecar file ``notebook_sources.json`` records, per notebook, the mtime
    of the .py source it was last generated from.
  * ``py_to_notebook`` only regenerates a notebook whose .py has changed since
    that recorded mtime (or which has no record). If the notebook is already
    up to date (recorded mtime == current .py mtime) it is skipped.
  * Before overwriting, the current notebook is backed up to
    ``notebook_backups/<name>.<epoch>.ipynb``.
"""

import re
import os
import json
import time
import nbformat as nbf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
    ("assemble_weekly_dataset.py",  "1_assemble_weekly_dataset.ipynb"),
    ("rf_regression.py",            "2_rf_regression.ipynb"),
    ("gam_regression.py",           "3_gam_regression.ipynb"),
    ("regression_ode_variant.py",   "4_regression_ode_variant.ipynb"),
    ("weekly_env_regression.py",    "2b_weekly_env_regression.ipynb"),
]

_SIDECAR = os.path.join(_SCRIPT_DIR, "notebook_sources.json")
_BACKUP_DIR = os.path.join(_SCRIPT_DIR, "notebook_backups")


def _load_sidecar():
    if os.path.exists(_SIDECAR):
        with open(_SIDECAR) as f:
            return json.load(f)
    return {}


def _save_sidecar(data):
    with open(_SIDECAR, "w") as f:
        json.dump(data, f, indent=2)


def _backup_notebook(nb_path):
    """Back up the current notebook before it is overwritten."""
    os.makedirs(_BACKUP_DIR, exist_ok=True)
    dest = os.path.join(_BACKUP_DIR, f"{os.path.basename(nb_path)}.{int(time.time())}.ipynb")
    with open(nb_path, "rb") as src:
        with open(dest, "wb") as out:
            out.write(src.read())
    return dest


def smart_split(py_path):
    """Split a .py file into logical notebook cells.

    Strategy: only split at top-level boundaries (before top-level def/class/
    decorator, or at double-blank-lines between top-level constructs). Nested
    functions and their bodies are never split.
    """
    with open(py_path, "r") as f:
        lines = f.readlines()

    cells = []

    def _indent(line):
        if line.strip() == "":
            return 0
        return len(line) - len(line.lstrip())

    def _is_top_level_def_or_class(line):
        s = line.strip()
        return _indent(line) == 0 and (s.startswith("def ") or s.startswith("class "))

    def _is_top_level_decorator(line):
        return _indent(line) == 0 and line.strip().startswith("@")

    def flush_code(chunk):
        src = "".join(chunk).rstrip("\n")
        if src.strip():
            cells.append(("code", src))

    def flush_md(chunk):
        src = "".join(chunk).rstrip("\n")
        if src.strip():
            cells.append(("markdown", src))

    i = 0
    n = len(lines)
    current_code = []
    depth = 0  # nesting depth: 0 = top level

    # ---- Module docstring at top -> markdown ----
    if lines and (lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''")):
        triple = lines[0].strip()[:3]
        doc_lines = [lines[0]]
        i = 1
        closed = triple in lines[0][1:]
        while i < n and not closed:
            doc_lines.append(lines[i])
            if triple in lines[i]:
                closed = True
            i += 1
        else:
            i += 1
        flush_md(doc_lines)

    # ---- Remaining lines: split into code cells ----
    while i < n:
        line = lines[i]
        stripped = line.strip()
        indent = _indent(line)

        # Track nesting depth via top-level def/class
        if _is_top_level_def_or_class(line):
            # If we were accumulating code, flush before starting new def
            # (handles case where blank lines between top-level functions were
            # not enough to trigger a split)
            pass

        # At top level (depth==0), check for cell-break opportunities
        if depth == 0:
            # Before a top-level def/class/decorator: flush and start new cell
            if (_is_top_level_def_or_class(line) or _is_top_level_decorator(line)):
                if current_code:
                    flush_code(current_code)
                    current_code = []
                # Accumulate this line and everything up to indent > 0
                current_code.append(line)
                i += 1
                # If this is a def/class, now we're entering a body
                if _is_top_level_def_or_class(line):
                    depth = 1
                continue

            # Blank line at top level: potential cell break
            if stripped == "":
                # Peek ahead past blanks
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1

                if j < n:
                    next_line = lines[j]
                    # Break before top-level def/class/decorator
                    if (_is_top_level_def_or_class(next_line)
                            or _is_top_level_decorator(next_line)):
                        flush_code(current_code)
                        current_code = []
                        i = j
                        continue
                    # Double blank at top level = cell break
                    if j - i >= 2:
                        flush_code(current_code)
                        current_code = []
                        i = j
                        continue

                current_code.append(line)
                i += 1
                continue

        # Inside a function body (or at top level but not a break point)
        current_code.append(line)

        # Track depth changes from indented def/class (nested)
        if stripped.startswith("def ") or stripped.startswith("class "):
            # We're entering a nested body
            depth += 1

        # Detect return to top level: next non-blank line has indent 0
        if stripped == "" or indent > 0:
            # Check if the NEXT non-blank line returns to depth 0
            if stripped == "":
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1
                if j < n and _indent(lines[j]) == 0:
                    depth = 0

        i += 1

    flush_code(current_code)
    return cells


def py_to_notebook(py_path, out_name=None):
    """Convert a Python script to a notebook, unless it is already up to date."""
    py_mtime = os.path.getmtime(py_path)
    out_path = os.path.join(
        os.path.dirname(py_path),
        out_name or os.path.basename(py_path).replace(".py", ".ipynb"),
    )

    sidecar = _load_sidecar()
    recorded = sidecar.get(os.path.basename(out_path))
    if recorded is not None and abs(recorded - py_mtime) < 1.0 and os.path.exists(out_path):
        print(f"  {os.path.basename(py_path)} -> {os.path.basename(out_path)}  "
              f"(up to date, skipped)")
        return

    # Safety: back up the current notebook before replacing it.
    if os.path.exists(out_path):
        backup = _backup_notebook(out_path)
        print(f"  backup: {os.path.basename(backup)}")

    raw_cells = smart_split(py_path)

    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    })

    for cell_type, source in raw_cells:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))

    # Post-process: fix __file__ and __main__ for Jupyter
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        # Fix __file__
        if "_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))" in cell.source:
            cell.source = cell.source.replace(
                "_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))",
                "try:\n    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\nexcept NameError:\n    _SCRIPT_DIR = os.getcwd()"
            )
        # Fix __main__ guard
        m = re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*\n\s+main\(\)', cell.source)
        if m:
            cell.source = cell.source[:m.start()] + "main()"

    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"  {os.path.basename(py_path)} -> {os.path.basename(out_path)}  "
          f"({len(nb.cells)} cells)")

    sidecar[os.path.basename(out_path)] = py_mtime
    _save_sidecar(sidecar)


def main():
    print("Converting .py -> .ipynb ...")
    for py_name, nb_name in TARGETS:
        py_path = os.path.join(_SCRIPT_DIR, py_name)
        if os.path.exists(py_path):
            py_to_notebook(py_path, nb_name)
        else:
            print(f"  SKIP (not found): {py_name}")
    print("Done.")


if __name__ == "__main__":
    main()
