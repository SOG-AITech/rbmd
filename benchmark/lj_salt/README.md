# LJ-salt benchmarks

RBMD configurations use RBL cutoff/core `8/4`, 30 neighbor samples, and 200
RBE samples.

- `workbook/run_LJ_salt.xlsx` contains the publication plotting data.
- `linear/rbmd/rbl8_rc4_n30_rbe200/` contains the main linear-complexity cases;
  `linear/rbmd/additional_sizes/` contains the two smaller LJ-salt sizes.
- `scaling/rbmd/rbl8_rc4_n30_rbe200/` contains strong- and weak-scaling cases.
- `rdf/rbl8_rc4_n30_rbe200/` contains RDF configurations and processed curves.
- `lammps_reference/pppm_cutoff8_reference/` contains the matched cutoff-8 PPPM
  reference set; `pppm_cutoff8_extended/` contains the additional large-device
  cases.

Water configurations are not duplicated in this system directory.
