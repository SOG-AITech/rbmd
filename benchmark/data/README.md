# RBMD publication data archive

The Release asset contains only the four input systems used for the published
RDF, linear-complexity, strong-scaling, and weak-scaling results. Large inputs
are not committed to the Git repository.

Download [`rbmd-data-v2.tar.gz`](https://github.com/qizhou1729/rbmd/releases/download/benchmark-data-v2/rbmd-data-v2.tar.gz)
and extract it from the repository root:

```bash
curl -L https://github.com/qizhou1729/rbmd/releases/download/benchmark-data-v2/rbmd-data-v2.tar.gz \
  -o rbmd-data-v2.tar.gz
tar -xzf rbmd-data-v2.tar.gz
```

| Installed path | System setup | SHA-256 |
| --- | --- | --- |
| `benchmark/data/h2o.data` | SPC/E water; 72,981 atoms; 24,327 molecules; 90 x 90 x 90 angstrom box | `220343ff4b38eea81f7a0be318721c88e890c25ffb34f6e640743ff259235a43` |
| `benchmark/data/h2o_rdf.data` | Equilibrated SPC/E water RDF seed; 2,703 atoms; 901 molecules; 30 x 30 x 30 angstrom box; replicated 3 x 3 x 3 for the 72,981-atom RDF case | `451693e9265aba1d7b3ba426c07f393767d813be58a4eb5ead783498166f5885` |
| `benchmark/data/lj_salt.data` | LJ electrolyte; 200,000 atoms; 100,000 cations and 100,000 anions; 271 x 271 x 271 reduced-unit box | `f715cf381f5e52cceb2afb84b7d5fa7c30da373a4c0db263bcc66757fb1c5fb5` |
| `benchmark/data/peo_litfsi.data` | PEO-LiTFSI; 976,200 atoms; 11,400 molecules; OPLS parameterization | `2be3937ea09aedc14a426ad835e7a0c901492cda5710251da6f91966740d8937` |

All benchmark and example configurations reuse these canonical paths. The
archive does not contain duplicate inputs, generated trajectories, job logs,
dated test directories, or intermediate analysis files.
