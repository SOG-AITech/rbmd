#!/usr/bin/env python3
"""Exercise the real GPU/MPI alpha gate; use a fresh output directory.

python3 tests/run_debug_alpha.py --binary /path/to/rbmd --output /path/to/new-dir --ranks 1
The caller supplies the MPI/toolchain/device environment. No NVE runs are made.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    binary, root = args.binary.resolve(), args.output.resolve()
    if not binary.is_file() or args.ranks < 1:
        parser.error("an existing binary and positive rank count are required")
    root.mkdir(parents=True, exist_ok=False)
    data = root / "ions.data"
    atoms = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                charge = 1 if (x + y + z) % 2 else -1
                atoms.append(f"{len(atoms)+1} {1 if charge > 0 else 2} {charge} "
                             f"{2+4*x+0.07*y} {2+4*y+0.03*z} {2+4*z+0.05*x} 0 0 0")
    data.write_text("Alpha gate fixture\n\n64 atoms\n2 atom types\n\n"
                    "0 16 xlo xhi\n0 16 ylo yhi\n0 16 zlo zhi\n\nMasses\n\n1 1\n2 1\n\n"
                    "Pair Coeffs\n\n1 1 1\n2 1 1\n\nAtoms # charge\n\n" + "\n".join(atoms) + "\n")
    base = {
        "init_configuration": {"read_data": {"file": str(data), "unit": "LJ",
            "atom_style": "charge", "velocity_type": "GAUSS", "replicate": [1, 1, 1]}},
        "hyper_parameters": {"force_field": {"type": "LJ/CUT/COUL/LONG"},
            "neighbor": {"type": "Verlet-List", "cut_off": 3, "r_core": 1.5,
                         "neighbor_sample_num": 16, "energy_rbl_flag": "no"},
            "extend": {"fix_shake": False}},
        "execution": {"ensemble": "NVT", "group": "all", "com_bias": "no", "integration_type": "vv",
            "par_a": 1, "par_b": 3, "temperature": [1, 1, 1], "temp_ctrl_type": "NOSE_HOOVER",
            "press_ctrl_type": "NOSE_HOOVER", "timestep": 0.001, "num_steps": 0,
            "diagnostics": {"fixed_samples": 3, "forces": True, "interval": 1, "prefix": "revision"}},
        "outputs": {"thermo_out": {"interval": 1}},
    }
    # Independent evaluation of the automatic real-space error estimate for this LJ fixture.
    auto_alpha = -math.log(1e-4 * math.sqrt(64 * 3 * 16**3) / (2 * 64)) / 3**2
    cases = [
        ("auto_default", {"accuracy": 1e-4}, None, auto_alpha),
        ("auto_false", {"accuracy": 1e-4, "debug_alpha": False}, None, auto_alpha),
        ("manual", {"debug_alpha": True, "alpha": 0.03}, None, 0.03),
        ("manual_precedence", {"debug_alpha": True, "alpha": 0.03, "accuracy": 1e-4}, None, 0.03),
        ("manual_matches_auto", {"debug_alpha": True, "alpha": auto_alpha}, None, auto_alpha),
        ("legacy_alpha", {"accuracy": 1e-4, "alpha": 0.03}, "requires debug_alpha=true", None),
        ("false_with_alpha", {"debug_alpha": False, "alpha": 0.03}, "requires debug_alpha=true", None),
        ("missing_alpha", {"debug_alpha": True, "accuracy": 1e-4}, "requires a numeric alpha", None),
        ("zero_alpha", {"debug_alpha": True, "alpha": 0}, "finite and > 0", None),
        ("negative_alpha", {"debug_alpha": True, "alpha": -1}, "finite and > 0", None),
        ("overflow_alpha", {"debug_alpha": True, "alpha": 1e300}, "finite and > 0", None),
        ("string_alpha", {"debug_alpha": True, "alpha": "0.03"}, "requires a numeric alpha", None),
        ("bool_alpha", {"debug_alpha": True, "alpha": True}, "requires a numeric alpha", None),
        ("string_flag", {"debug_alpha": "true", "alpha": 0.03}, "must be a boolean", None),
        ("integer_flag", {"debug_alpha": 1, "alpha": 0.03}, "must be a boolean", None),
        ("missing_accuracy", {}, "Coulomb accuracy must be defined", None),
    ]
    results = []
    for solver in ("RBE", "EWALD"):
        forces = {}
        for name, options, error, expected_alpha in cases:
            path = root / f"{solver.lower()}_{name}"
            path.mkdir()
            config = copy.deepcopy(base)
            config["hyper_parameters"]["coulomb"] = dict(
                type=solver, coulomb_sample_num=32, energy_rbe_flag="no", kmax=[4, 4, 4], **options)
            if solver == "RBE":
                config["hyper_parameters"]["coulomb"].update(rbe_low_modes=True, K_max=0, random_seed=12345)
            (path / "run.json").write_text(json.dumps(config, indent=2))
            with (path / "run.log").open("w") as log:
                run = subprocess.run(["mpirun", "-np", str(args.ranks), str(binary), "-j", str(path / "run.json")],
                                     cwd=path, stdout=log, stderr=subprocess.STDOUT, timeout=90)
            log = (path / "run.log").read_text(errors="replace")
            if error:
                assert run.returncode != 0 and error in log, (path, run.returncode, error)
            else:
                assert run.returncode == 0, (path, run.returncode)
                source = re.search(r"alpha_source\s*:\s*(\w+)", log).group(1)
                assert source == ("manual" if options.get("debug_alpha") else "auto"), path
                alpha = float(re.search(r"\balpha\s*:\s*([\d.eE+-]+)", log).group(1))
                assert math.isclose(alpha, expected_alpha, rel_tol=2e-6), (path, alpha, expected_alpha)
                samples = {}
                for shard in path.glob("revision.forces.rank*.csv"):
                    with shard.open() as f:
                        for row in csv.DictReader(f):
                            key = (int(row["sample"]), int(row["id"]))
                            assert key not in samples, (path, key)
                            samples[key] = tuple(float(row[k]) for k in ("fx", "fy", "fz"))
                assert len(samples) == 3 * 64 and {s for s, _ in samples} == {1, 2, 3}, path
                assert all(math.isfinite(v) for force in samples.values() for v in force), path
                forces[name] = samples
            results.append({"solver": solver, "case": name, "exit_code": run.returncode, "passed": True})
            print(f"PASS {solver} {name} ranks={args.ranks}", flush=True)
        for left, right in [("auto_default", "auto_false"), ("manual", "manual_precedence"),
                            ("auto_default", "manual_matches_auto")]:
            assert forces[left].keys() == forces[right].keys()
            assert all(math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-6)
                       for key in forces[left] for a, b in zip(forces[left][key], forces[right][key])), (solver, left, right)
    (root / "PASS.json").write_text(json.dumps({"binary": str(binary),
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "ranks": args.ranks,
        "fixed_samples": 3, "atoms": 64, "cases": results, "force_equivalence_pairs": 6}, indent=2) + "\n")


if __name__ == "__main__":
    main()
