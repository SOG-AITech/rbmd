# RBMD data archive

Large inputs are distributed as one GitHub Release asset and are not committed
to the Git repository:

[`rbmd-data-v1.tar.gz`](https://github.com/SOG-AITech/rbmd/releases/download/benchmark-data-v1/rbmd-data-v1.tar.gz)

From the repository root, download and extract the archive, then make sure the
inputs are installed at the canonical paths in the table below:

```bash
curl -L https://github.com/SOG-AITech/rbmd/releases/download/benchmark-data-v1/rbmd-data-v1.tar.gz \
  -o rbmd-data-v1.tar.gz
tar -xzf rbmd-data-v1.tar.gz
```

| Installed path | Purpose | Approximate size |
| --- | --- | ---: |
| `benchmark/data/h2o.data` | H2O scaling and linear benchmarks | 13 MB |
| `benchmark/data/h2o_rdf.data` | Equilibrated H2O RDF input before 3x3x3 replication | 0.55 MB |
| `benchmark/data/lj_salt.data` | LJ-salt benchmarks | 28 MB |
| `benchmark/data/peo_litfsi.data` | PEO-LiTFSI 1M base input | 332 MB |
| `examples/ljsalt/lj_salt_20w_v.data` | Large LJ-salt example | 28 MB |

If an existing copy of the data asset uses an older internal path, move the
equilibrated water input to `benchmark/data/h2o_rdf.data`. Scaling cases reuse
the shared system inputs instead of storing additional copies.
