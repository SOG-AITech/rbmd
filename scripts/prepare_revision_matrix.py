#!/usr/bin/env python3
"""Generate force-only ablation inputs from a working RBMD JSON/JSONC input.
No simulation is launched. EWALD kmax must be converged independently.
"""
import argparse
import copy
import json
import math
import re
from pathlib import Path


def load_jsonc(path):
    # Preserve quoted strings (including URLs) while removing JSON comments.
    token = r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/'
    text = re.sub(token, lambda m: m.group(0) if m.group(0).startswith('"') else ' ', path.read_text())
    return json.loads(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--debug-alpha", action="store_true", help="Enable manual alpha for debugging only")
    p.add_argument("--alpha", type=float, help="Manual alpha=g_ewald^2; requires --debug-alpha")
    p.add_argument("--kmax", type=int, nargs=3, required=True, help="Candidate reference cutoff; caller must verify convergence")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--rbe-samples", type=int, required=True)
    p.add_argument("--rbl-samples", type=int, required=True)
    args = p.parse_args()
    if args.debug_alpha != (args.alpha is not None):
        p.error("--debug-alpha and --alpha must be supplied together")
    if args.alpha is not None and (not math.isfinite(args.alpha) or args.alpha <= 0):
        p.error("alpha must be finite and positive")
    if min(args.kmax + [args.samples, args.rbe_samples, args.rbl_samples]) <= 0:
        p.error("all sample counts and kmax components must be positive")
    base = load_jsonc(args.input)
    hp = base["hyper_parameters"]
    if hp["force_field"]["type"] not in ("CVFF", "LJ/CUT/COUL/LONG"):
        p.error("four-way RBE/RBL ablation requires CVFF or LJ/CUT/COUL/LONG")
    if "r_core" not in hp["neighbor"]:
        p.error("base input must provide neighbor.r_core")
    coulomb = hp.setdefault("coulomb", {})
    if not args.debug_alpha:
        if "alpha" in coulomb or coulomb.get("debug_alpha", False) is not False:
            p.error("remove manual alpha/debug_alpha from the base input or supply --debug-alpha --alpha")
        accuracy = coulomb.get("accuracy")
        if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or accuracy <= 0:
            p.error("automatic alpha requires a finite positive coulomb.accuracy in the base input")
    else:
        coulomb.update(debug_alpha=True, alpha=args.alpha)
    data = Path(base["init_configuration"]["read_data"]["file"])
    if not data.is_absolute():
        data = (args.input.resolve().parent / data).resolve()
    base["init_configuration"]["read_data"]["file"] = str(data)
    cases = [("reference_candidate", "Verlet-List", "EWALD", 1),
             ("rbe_only", "Verlet-List", "RBE", args.samples),
             ("rbl_only", "RBL", "EWALD", args.samples),
             ("rbl_rbe", "RBL", "RBE", args.samples)]
    # Refuse overwrites: each run must preserve its raw samples and metadata.
    if args.output.exists():
        p.error("output directory already exists; use a new directory")
    args.output.mkdir(parents=True)
    for name, neighbor, coulomb, samples in cases:
        cfg = copy.deepcopy(base)
        cfg["hyper_parameters"]["neighbor"].update(type=neighbor, neighbor_sample_num=args.rbl_samples, energy_rbl_flag="no")
        cfg["hyper_parameters"]["coulomb"].update(type=coulomb,
                  kmax=args.kmax, coulomb_sample_num=args.rbe_samples, energy_rbe_flag="no")
        cfg["execution"]["num_steps"] = 0
        cfg["execution"]["diagnostics"] = dict(fixed_samples=samples, interval=1, forces=True,
                 energy=False, thermostat_heat=False, isolated_timing=True, prefix="revision")
        case = args.output / name
        case.mkdir()
        (case / "rbmd.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (args.output / "README.txt").write_text(
        "Run each rbmd.json from its own directory using the same executable, MPI layout and environment.\n"
        "reference_candidate is NOT certified converged. Increase kmax and verify reference force convergence.\n"
        "Fixed configurations: no integration or thermostat; sample indices advance the RBL epoch.\n"
        "Alpha is automatic unless --debug-alpha --alpha was explicitly supplied. RBE sampling is not changed here.\n"
        "Inspect force_statistics.json and metadata; do not equate isolated timing with production throughput.\n")
    print(args.output)


if __name__ == "__main__":
    main()
