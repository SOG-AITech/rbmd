# Benchmarks

This directory contains publication-oriented configurations, RDF results, and
plotting workbooks for three systems:

- `h2o/`: water RDF, linear-complexity, strong-scaling, and weak-scaling cases.
- `lj_salt/`: LJ-salt RBMD cases and matched LAMMPS-Kokkos references.
- `peo_litfsi/`: PEO-LiTFSI timing and RDF cases.
- `data/`: canonical locations for separately distributed input data.

## Naming convention

Directory names use lowercase `snake_case` and identify the system, method,
scientific role, parameter set, or device count. They do not encode collection
dates. Device-specific cases use names such as `dcu_04`; atom-count cases use
names such as `atoms_00583848`.

## Running a case

Build RBMD by following the [build guide](../docs/build.en.md), install the
inputs described in [`data/README.md`](data/README.md), and review the Slurm
resources for the target cluster. Submit a case beside its job file, for
example:

```bash
cd benchmark/h2o/scaling/rbmd/strong/dcu_04
sbatch run.slurm
```

Absolute paths retained in manifests or metadata are provenance records from
the source runs. Portable H2O scripts resolve the repository checkout at run
time and allow `RBMD_BIN` and `ROCM_PATH` overrides.
