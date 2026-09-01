# RBMD data archive

Large inputs are distributed as one GitHub Release asset and are not committed
to the Git repository:

[`rbmd-data-v1.tar.gz`](https://github.com/SOG-AITech/rbmd/releases/download/benchmark-data-v1/rbmd-data-v1.tar.gz)

From the repository root, download and extract the archive:

```bash
curl -L https://github.com/SOG-AITech/rbmd/releases/download/benchmark-data-v1/rbmd-data-v1.tar.gz \
  -o rbmd-data-v1.tar.gz
tar -xzf rbmd-data-v1.tar.gz
```

The archive preserves repository-relative paths and contains:

| Installed path | Purpose | Approximate size |
| --- | --- | ---: |
| `benchmark/data/h2o.data` | H2O scaling and linear benchmarks | 13 MB |
| `benchmark/data/lj_salt.data` | LJ-salt benchmarks | 28 MB |
| `benchmark/data/peo_litfsi.data` | PEO-LiTFSI 1M base input | 332 MB |
| `benchmark/h2o/rdf/rdf_compare_final_100k_20260731/equi_bulk.4000000.data` | H2O RDF benchmark | 0.55 MB |
| `examples/ljsalt/lj_salt_20w_v.data` | Large LJ-salt example | 28 MB |

Scaling cases reuse the corresponding shared system input instead of storing
additional copies.
