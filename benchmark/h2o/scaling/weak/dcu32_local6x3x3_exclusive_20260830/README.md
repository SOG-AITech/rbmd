# H2O weak-scaling case

- Source record: /public/home/acymkxdx47/tmp_rbmd_h2o_scaling_pbc_only_1f34_9_5p5_n50_p200_20260830/weak/cases/dcu32_local6x3x3_exclusive_20260830
- Resources: 8 nodes, 32 MPI ranks/DCUs, exclusive allocation, RBMD_MPI_GRID=4x4x2.
- Local replication: 24x12x6; force field: CVFF; neighbor: RBL cutoff/core 9.0/5.5, sample count 50; Coulomb: RBE sample count 200.
- Input data is shared from benchmark/data/h2o.data.
- Set RBMD_BIN to override the default RBMD2 build/rbmd executable.

Only the reusable configuration and Slurm script are retained. The copied package intentionally excludes rbmd_buildsync, rbmd.log, Slurm logs, runtime metadata, temperature.txt, and thermo.txt.
