# v5 GPU core-coverage regression

Copied unchanged from the qualified v5 GPU fixture. The runner verifies production-source
hashes, extracts the actual HIP kernels and wrappers, and compiles them using this build's
compiler flags. It fails if no real GPU is available.

Run inside a Slurm GPU allocation with the RBMD runtime loaded:

```sh
python3 tests/rbl_fixedp_v5/build_and_run.py --source "$PWD" --build "$PWD/build" --output /absolute/path/to/a/new-test-directory
```

The fixture checks producer padding boundaries, missing physical/padding core entries,
duplicate core entries, periodic alias equivalence, wrong force direction, changed charge/type,
and witness-buffer bounds. It does not establish RDF, force RMS, throughput, or complete CVFF
special-force GPU coverage. Its source pins deliberately reject unreviewed algorithm changes.
