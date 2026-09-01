"""Apply all Jupyter-specific fixes to the numbered notebooks.

Run AFTER convert_to_notebooks.py. Fixes applied:
1. Replace matplotlib.use(\"Agg\") with %matplotlib inline
2. Replace _SCRIPT_DIR resolution with a Jupyter-safe version (search for regression dir)
3. Replace the `if __name__ == \"__main__\":` guard with a direct main() call
4. Replace fig.savefig(...) with plt.show(), dropping _FIG_DIR/fig-path plumbing
5. (assemble notebook) Clip b1_eff/b2_eff to [0,1]
6. Ensure ```import os``` is available where _SCRIPT_DIR is defined

Guard
-----
Fixes are only applied to a notebook that was freshly regenerated from the
current .py (i.e. the sidecar written by convert_to_notebooks.py records the
current .py mtime for it). A notebook that is newer than its .py and does NOT
match the sidecar has been manually edited, so it is left untouched.
"""

import os
import re
import json
import nbformat as nbf

BASE = os.path.dirname(os.path.abspath(__file__))

NOTEBOOKS = [
    "1_assemble_weekly_dataset.ipynb",
    "2_rf_regression.ipynb",
    "2b_weekly_env_regression.ipynb",
    "3_gam_regression.ipynb",
    "4_regression_ode_variant.ipynb",
]

_SIDECAR = os.path.join(BASE, "notebook_sources.json")

# Map each notebook to the .py it was generated from (matches convert_to_notebooks)
_SOURCES = {
    "1_assemble_weekly_dataset.ipynb": "assemble_weekly_dataset.py",
    "2_rf_regression.ipynb": "rf_regression.py",
    "2b_weekly_env_regression.ipynb": "weekly_env_regression.py",
    "3_gam_regression.ipynb": "gam_regression.py",
    "4_regression_ode_variant.ipynb": "regression_ode_variant.py",
}


def _load_sidecar():
    if os.path.exists(_SIDECAR):
        with open(_SIDECAR) as f:
            return json.load(f)
    return {}

# Jupyter-safe path block (locates the regression dir by searching downward)
PATH_BLOCK = (
    "try:\n"
    "    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n"
    "except NameError:\n"
    "    # In Jupyter there is no __file__. Search downward for the 'regression'\n"
    "    # folder that contains this notebook family (it has results/ and data/\n"
    "    # siblings), so paths resolve regardless of the kernel launch directory.\n"
    "    import pathlib\n"
    "    _cwd = pathlib.Path(os.getcwd()).resolve()\n"
    "    _reg = None\n"
    "    for _p in _cwd.rglob(\"regression\"):\n"
    "        if not _p.is_dir():\n"
    "            continue\n"
    "        if (_p / \"results\").is_dir() and (_p / \"data\").is_dir():\n"
    "            _reg = _p\n"
    "            break\n"
    "    if _reg is None:\n"
    "        # Fallback: any regression dir under cwd\n"
    "        for _p in _cwd.rglob(\"regression\"):\n"
    "            if _p.is_dir():\n"
    "                _reg = _p\n"
    "                break\n"
    "    _SCRIPT_DIR = str(_reg) if _reg is not None else str(_cwd)"
)


def fix_code_cell(cell, notebook):
    src = cell.source

    # --- 1. Agg -> inline ---
    src = src.replace('matplotlib.use("Agg")', "%matplotlib inline")

    # --- 2. Path block ---
    # Replace any existing try:/except:NameError _SCRIPT_DIR block with the
    # Jupyter-safe version.
    if "_SCRIPT_DIR" in src and "os.path.dirname(os.path.abspath(__file__))" in src:
        # Find start of the try:
        start = src.find("try:")
        if start != -1:
            after = src[start:]
            lines = after.split("\n")
            # Find the end of the except clause: the last line that assigns
            # _SCRIPT_DIR inside the except (either str(_reg) or os.getcwd() or
            # str(_cwd)).
            end_rel = None
            for k, ln in enumerate(lines):
                if re.search(r"_SCRIPT_DIR\s*=\s*(str\(_reg\)|str\(_cwd\)|os\.getcwd\(\))", ln):
                    end_rel = len("\n".join(lines[: k + 1]))
            if end_rel is not None:
                src = src[:start] + PATH_BLOCK + src[start + end_rel :]

    # Ensure `import os` precedes the path block (in case this cell is standalone)
    if "try:" in src and "import os" not in src.split("try:")[0] and "import os" not in src:
        src = "import os\n\n" + src

    # --- 3. __main__ guard -> main() ---
    m = re.search(r"if\s+__name__\s*==\s*[\"\']__main__[\"\']\s*:\s*\n\s+main\(\)", src)
    if m:
        src = src[: m.start()] + "main()"

    # --- 4. savefig -> show, drop FIG plumbing ---
    # Handles multi-line savefig(...) calls: when a savefig line is dropped we
    # must also drop its continuation argument lines (indented lines up to the
    # closing parenthesis), otherwise an orphaned "dpi=..., bbox_inches=...)" is
    # left behind producing invalid indentation.
    lines = src.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if re.search(r"_FIG_DIR\s*=\s*os\.path\.join", s):
            i += 1
            continue
        if re.search(r"os\.makedirs\(_FIG_DIR", s):
            i += 1
            continue
        if re.search(r"^\s*(path|fig_path)\s*=\s*os\.path\.join\(_FIG_DIR", s):
            i += 1
            continue
        is_savefig = ("fig.savefig(" in s) or ("plt.savefig(" in s)
        if is_savefig:
            new_lines.append(" " * (len(ln) - len(ln.lstrip())) + "plt.show()")
            i += 1
            # Drop continuation lines of a multi-line savefig call: skip
            # indented lines until the call's closing parenthesis is seen.
            if not s.endswith(")"):
                depth = s.count("(") - s.count(")")
                while i < len(lines):
                    cont = lines[i]
                    if cont.strip() == "":
                        i += 1
                        continue
                    depth += cont.count("(") - cont.count(")")
                    if depth <= 0 and ")" in cont:
                        i += 1
                        break
                    i += 1
            continue
        if re.search(r"print\(f?[\"\'].*Saved", s) and ("path" in s or "fig_path" in s):
            i += 1
            continue
        new_lines.append(ln)
        i += 1
    src = "\n".join(new_lines)

    # --- 5. Clip b1_eff/b2_eff (assemble notebook) ---
    if "b1_eff = bm_fitted[w] / a_weekly" in src:
        clip_code = (
            "\n\n            # b1 and b2 are per-bite infection probabilities, so they are\n"
            "            # physically constrained to [0, 1]. The weekly free-beta fit can\n"
            "            # produce beta_m spikes (esp. early/late year) that inflate b1\n"
            "            # beyond 1; clip to enforce the biological bound.\n"
            "            b1_eff = np.clip(b1_eff, 0.0, 1.0) if np.isfinite(b1_eff) else np.nan\n"
            "            b2_eff = np.clip(b2_eff, 0.0, 1.0) if np.isfinite(b2_eff) else np.nan"
        )
        src = src.replace(
            "            b1_eff = bm_fitted[w] / a_weekly if a_weekly > 0 else np.nan\n"
            "            b2_eff = bh_fitted[w] / a_weekly if a_weekly > 0 else np.nan",
            "            b1_eff = bm_fitted[w] / a_weekly if a_weekly > 0 else np.nan\n"
            "            b2_eff = bh_fitted[w] / a_weekly if a_weekly > 0 else np.nan"
            + clip_code,
        )

    cell.source = src


def main():
    sidecar = _load_sidecar()
    for name in NOTEBOOKS:
        path = os.path.join(BASE, name)
        py_name = _SOURCES.get(name)
        py_path = os.path.join(BASE, py_name) if py_name else None

        # Guard: only fix if this notebook was regenerated from the current .py.
        if py_name and os.path.exists(path) and os.path.exists(py_path):
            py_mtime = os.path.getmtime(py_path)
            recorded = sidecar.get(name)
            if recorded is None or abs(recorded - py_mtime) >= 1.0:
                print(f"{name}: SKIP (not freshly regenerated from current .py; "
                      f"leaving any manual edits untouched)")
                continue

        nb = nbf.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                fix_code_cell(cell, name)
        nbf.write(nb, path)
        n_code = sum(1 for c in nb.cells if c.cell_type == "code")
        print(f"{name}: processed ({n_code} code cells)")
    print("Done.")


if __name__ == "__main__":
    main()
