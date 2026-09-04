#!/usr/bin/env python3
"""Generate an RBMD parameter sweep.

Reads:
  - template.json       (base config with {{PLACEHOLDER}} tokens)
  - sweep_params.json   (which params are fixed vs. swept this round)

Writes, for every combination of the swept parameters:
  - runs/<tag>/test_rshell.json   fully rendered config for that combo
  - runs/<tag>/params.json        the exact parameter values used (for bookkeeping)
  - runs/<tag>/structure          a symlink to the real structure directory,
                                  so the relative path in the json never breaks
                                  no matter how deep runs/<tag> is nested.

Also writes sweep_table.tsv (array_index<TAB>tag), which run_sweep.slurm uses
to figure out which combo each Slurm array task should run.

Usage:
    python gen_sweep.py

Then submit the whole sweep as ONE job (see printed sbatch command at the end).

If your real structure/ directory is not the sibling of this script's parent
(i.e. not "../structure"), edit STRUCTURE_DIR below once.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

TEMPLATE_PATH = Path("template.json")
SWEEP_PARAMS_PATH = Path("sweep_params.json")
OUT_TABLE = Path("sweep_table.tsv")
RUNS_DIR = Path("runs")

# Adjust this once if your structure/ folder lives somewhere else relative
# to the directory you run this script from (that directory should be the
# same one that holds run_sweep.slurm, template.json, sweep_params.json).
STRUCTURE_DIR = Path("../structure").resolve()

# sweep_params.json key -> template placeholder token
PLACEHOLDERS = {
    "neighbor_type": "{{NEIGHBOR_TYPE}}",
    "cut_off": "{{CUTOFF}}",
    "r_core": "{{RCORE}}",
    "neighbor_sample_num": "{{NSN}}",
    "coulomb_type": "{{COULOMB_TYPE}}",
    "coulomb_sample_num": "{{CSN}}",
}

# Short prefixes used to build human-readable directory/tag names
TAG_ABBR = {
    "neighbor_type": "nb",
    "cut_off": "co",
    "r_core": "rc",
    "neighbor_sample_num": "nsn",
    "coulomb_type": "cl",
    "coulomb_sample_num": "csn",
}


def main() -> None:
    sweep = json.loads(SWEEP_PARAMS_PATH.read_text(encoding="utf-8"))
    sweep.pop("_comment", None)

    missing = set(PLACEHOLDERS) - set(sweep)
    if missing:
        raise SystemExit(f"sweep_params.json is missing keys: {sorted(missing)}")
    extra = set(sweep) - set(PLACEHOLDERS)
    if extra:
        raise SystemExit(f"sweep_params.json has unknown keys: {sorted(extra)}")

    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    keys = list(PLACEHOLDERS.keys())
    value_lists = [sweep[k] for k in keys]
    for k, vals in zip(keys, value_lists):
        if not isinstance(vals, list) or len(vals) == 0:
            raise SystemExit(f"sweep_params.json[{k!r}] must be a non-empty list")

    varying_keys = [k for k, vals in zip(keys, value_lists) if len(vals) > 1]

    if not STRUCTURE_DIR.is_dir():
        print(f"warning: STRUCTURE_DIR={STRUCTURE_DIR} does not exist; "
              f"edit STRUCTURE_DIR at the top of this script.")

    RUNS_DIR.mkdir(exist_ok=True)
    rows: list[tuple[str, dict]] = []
    seen_tags: set[str] = set()

    for combo in itertools.product(*value_lists):
        combo_dict = dict(zip(keys, combo))

        tag_parts = [f"{TAG_ABBR[k]}{combo_dict[k]}" for k in varying_keys]
        tag = "_".join(tag_parts) if tag_parts else "baseline"
        if tag in seen_tags:
            raise SystemExit(f"duplicate tag generated: {tag} (values collide?)")
        seen_tags.add(tag)

        run_dir = RUNS_DIR / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        rendered = template_text
        for k, placeholder in PLACEHOLDERS.items():
            rendered = rendered.replace(placeholder, str(combo_dict[k]))
        if "{{" in rendered:
            raise SystemExit(f"template.json still has an unfilled placeholder for combo {combo_dict}")

        (run_dir / "test_rshell.json").write_text(rendered, encoding="utf-8")
        (run_dir / "params.json").write_text(json.dumps(combo_dict, indent=2), encoding="utf-8")

        struct_link = run_dir / "structure"
        if struct_link.is_symlink() or struct_link.exists():
            struct_link.unlink()
        if STRUCTURE_DIR.is_dir():
            struct_link.symlink_to(os.path.relpath(STRUCTURE_DIR, run_dir))

        rows.append((tag, combo_dict))

    with OUT_TABLE.open("w", encoding="utf-8") as fh:
        for idx, (tag, _) in enumerate(rows):
            fh.write(f"{idx}\t{tag}\n")

    n = len(rows)
    print(f"generated {n} run directories under {RUNS_DIR}/")
    print(f"wrote lookup table: {OUT_TABLE}")
    print()
    print("submit the whole sweep as ONE job array (example: cap 6 running at once):")
    print(f"  sbatch --array=0-{n - 1}%6 run_sweep.slurm")


if __name__ == "__main__":
    main()
