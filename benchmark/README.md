# Benchmarks

This directory contains curated configurations, submission scripts, RDF
results, and workbooks for the H2O, LJ-salt, and PEO-LiTFSI systems.

- `data/`: expected locations and descriptions for the separately distributed
  large input datasets.
- `h2o/`: water benchmarks and `workbook/run_H2O.xlsx`.
- `lj_salt/`: LJ-salt benchmarks and `workbook/run_LJ_salt.xlsx`.
- `peo_litfsi/`: polymer electrolyte benchmarks and `workbook/run_PEO.xlsx`.

## Before running

Build RBMD by following the [build guide](../docs/build.en.md), then place the
separately distributed inputs at the paths described in
[`data/README.md`](data/README.md). The configuration format is documented in
the [JSON configuration reference](../docs/json-configuration.en.md).

The multi-device campaigns target HPC environments. Before submitting a case,
review its JSON configuration and Slurm script, and adapt the following items:

- RBMD or LAMMPS executable and input-data paths;
- `ROOT` and other absolute paths retained from the original campaign;
- Slurm account, partition, node, task, GPU/DCU, and time requests;
- compiler/MPI modules, launcher options, and device environment variables.

The archived settings describe the original runs and are not portable cluster
defaults.

## Slurm and shell scripts

Each campaign contains one or more job files named `run.slurm`, `rbmd.slurm`,
or `lammps.slurm`. Submit a single reviewed case from its directory, for
example:

```bash
cd benchmark/<system>/<campaign>/cases/<case>
sbatch run.slurm
```

Some older campaigns use a different directory depth or place the Slurm file
directly in a numbered GPU directory. In those cases, run `sbatch` beside the
corresponding `.slurm` file.

Campaign-level shell helpers should be run from the campaign directory after
their `ROOT`, manifest paths, and Slurm settings have been reviewed:

```bash
cd benchmark/<system>/<campaign>
bash scripts/submit_remaining.sh
```

The exact helper name and location vary by campaign:

- `submit_all*.sh`, `submit_remaining.sh`, `submit_once.sh`: submit all,
  unsubmitted, or one configured case and record the Slurm job ID.
- `wait_and_submit.sh`: submit cases gradually while respecting queue limits
  and dependencies.
- `status.sh`: summarize submitted, queued, and validated cases.
- `build_preload.sh`: build the optional MPI-topology preload library required
  by the associated topology-control campaign.
- `archive_campaign.sh`: package a completed campaign; it is optional and is
  not required to reproduce a benchmark run.

Submission helpers may wait and poll the Slurm queue. Use the individual
`.slurm` files when automatic batch submission is not appropriate for the
target system.

## Python utilities

The Python files are support and analysis utilities:

- `collect_results.py` parses completed run output and produces campaign
  summary tables.
- `prepare_campaign.py` and `prepare_water_same_sizes_shape_campaign.py`
  generate campaign directories, configurations, manifests, and Slurm files
  from the retained templates.
- `gen_sweep.py` generates the PEO-LiTFSI RDF parameter-sweep cases.
- `analyze_sweep_rdf.py` computes and compares RDF curves across that sweep.
- `parallel_rdf.py` calculates partial RDFs from trajectory frames in parallel.
- `trajectory_to_data.py` converts the final RBMD trajectory frame to a
  charge-style LAMMPS data file.

When a shell or Slurm script already calls one of these utilities, no separate
Python step is needed.
