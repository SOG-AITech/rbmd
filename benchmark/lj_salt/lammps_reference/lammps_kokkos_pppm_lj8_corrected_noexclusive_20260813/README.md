# Corrected LJ-salt cutoff 8/8

- Base data: /public/home/acymkxdx47/rbmd/build/runljsalt/scaling_rbl5_rc2p5_n200_rbe30_4m_per_dcu_20260804/lj_salt_20w_v.data (200,000 atoms)
- Strong scaling: 28,000,000 atoms, 1/2/4/8 nodes
- Weak scaling: 28,000,000 atoms per node, 1/2/4/8 nodes
- Linear complexity: 3.6M, 7.2M, 14.4M on 1 node; reuse strong-1N for 28M
- Four DCUs and four MPI ranks per node
- PPPM 1e-4, newton off, timer full sync, HIP_LAUNCH_BLOCKING=1, 2,000 steps
- No Slurm script contains --exclusive
- A result is valid only after pre-run and post-run atom-count checks pass
