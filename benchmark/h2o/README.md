# Water data package

- The current primary RBMD weak-scaling case is under scaling/weak/dcu32_local6x3x3_exclusive_20260830.
- It uses CVFF, periodic bond replication, RBL cutoff/core 9.0/5.5 with 50 neighbor samples, and RBE with 200 Coulomb samples.
- The scaling and linear cases use the shared `benchmark/data/h2o.data` input.
- The RDF case uses `rdf/rdf_compare_final_100k_20260731/equi_bulk.4000000.data`; both inputs are supplied by the separate data archive.
- The final workbook remains `workbook/run_H2O.xlsx`; Water RDF outputs remain under `rdf/`.
- Historical linear-complexity, topology-control, and matched LAMMPS configurations remain in their respective directories.
- Build products and runtime logs are intentionally excluded from the publication tree.
