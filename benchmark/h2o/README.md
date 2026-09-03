# H2O benchmarks

This package contains the publication H2O configurations and results for RDF,
linear complexity, strong scaling, and weak scaling. Directory names describe
the scientific role of each case and do not encode a campaign date.

## Method settings

All RBMD cases use CVFF with periodic bond replication, an uppercase `RBL`
neighbor method (`cut_off = 9.0`, `r_core = 5.5`, 50 neighbor samples), and
`RBE` electrostatics (200 Coulomb samples, accuracy `1e-4`). The timestep is
0.5 fs. Timing cases run for 1,000 steps; the RDF case runs for 20,000 steps.
Multi-device cases use four DCUs per node.

## Contents

- `rdf/`: the matched 150-bin LAMMPS/RBMD RDF curves, peak comparison, and the
  20,000-step RBMD input configuration.
- `linear/rbmd/`: six one-node RBMD cases and their timing manifest.
- `scaling/rbmd/strong/`: five fixed-size cases from 4 to 64 DCUs.
- `scaling/rbmd/weak/`: five fixed-work-per-DCU cases from 4 to 64 DCUs.
- `workbook/run_H2O.xlsx`: the plotting workbook used for the publication data.

Each timing manifest reports the short-range time, long-range time, and their
sum in seconds per step. The sum is the RBMD force-kernel total used for the
component comparison; it is not labeled as whole-program wall time.

## RDF check

The RDF comparison uses 150 bins with a 0.06 A bin width and a 9.0 A maximum
radius. All three principal peak positions agree exactly on that grid. The
RBMD/LAMMPS principal-peak height differences are 2.59% for O-O, 2.00% for
O-H, and 1.25% for H-H. Detailed values and whole-curve errors are in
`rdf/rdf_peak_summary.csv`.

The recorded server jobs referenced by the manifests completed successfully
with exit code zero. Exact job IDs and source record paths are retained in the
TSV files for provenance without exposing runtime logs or temporary tooling in
this publication package.
