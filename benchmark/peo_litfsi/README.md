# PEO-LiTFSI benchmarks

- `workbook/run_PEO.xlsx` contains the publication plotting data.
- `linear/rbmd/` contains the RBMD linear-complexity configurations.
- `scaling/strong/` contains the fixed-size scaling cases, named by DCU count.
- `scaling/weak/reference/` contains the fixed-work-per-DCU cases.
- `scaling/weak/validation/` contains the documented grid and repeatability
  checks for the first weak-scaling point.
- `rdf/rbl12_rc6_n200_rbe200/` contains the RDF/CN configurations and bin-width
  analyses for RBL cutoff/core 12/6, 200 neighbor samples, and 200 RBE samples.

RDF bin directories use stable names such as `bin_0p30`. Large input and
trajectory files remain outside the Git repository.
