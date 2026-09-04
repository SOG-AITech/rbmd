# LJ-salt LAMMPS-Kokkos reference

- Base input: 200,000 atoms.
- Strong scaling: 28,000,000 atoms on 1, 2, 4, and 8 nodes.
- Weak scaling: 28,000,000 atoms per node on 1, 2, 4, and 8 nodes.
- Linear complexity: 3.6M, 7.2M, and 14.4M atoms on one node; the 28M point is
  shared with the one-node strong-scaling case.
- Four DCUs and four MPI ranks are used per node.
- PPPM accuracy is `1e-4`, Kokkos uses `newton off`, full timer synchronization
  is enabled, and each case runs for 2,000 steps.

A result is accepted only when both pre-run and post-run atom-count checks pass.
