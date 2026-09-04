#!/usr/bin/env python3
from pathlib import Path
import csv
import re

root = Path('/public/home/acymkxdx47/rbmd/build/lammps_kokkos_pppm_lj8_corrected_noexclusive_20260813')
submitted = {}
with (root / 'submitted_jobs.tsv').open(newline='') as handle:
    for row in csv.DictReader(handle, delimiter='	'):
        submitted[row['case_dir']] = row['job_id']

rows = []
with (root / 'all_target_cases.tsv').open(newline='') as handle:
    for target in csv.DictReader(handle, delimiter='	'):
        case = Path(target['case_dir'])
        runtime = case / 'runtime_metadata.txt'
        log = case / 'log.lammps'
        text = log.read_text(errors='replace') if log.exists() else ''
        actual_match = re.findall(r'Loop time of\s+[0-9.eE+-]+\s+on\s+\d+\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms', text)
        def section(name):
            match = re.search(rf'(?m)^{name}\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)', text)
            return tuple(map(float, match.groups())) if match else None
        pair = section('Pair')
        kspace = section('Kspace')
        actual = int(actual_match[-1][1]) if actual_match else None
        valid = (case / 'VALIDATED_COMPLETE').exists() and actual == int(target['atoms']) and pair and kspace
        status = 'complete' if valid else ('invalid_atom_count' if actual is not None and actual != int(target['atoms']) else ('failed' if 'ERROR:' in text else 'missing_or_incomplete'))
        row = dict(target)
        row.update(job_id=submitted.get(str(case), ''), status=status, actual_atoms=actual or '',
                   pair_avg_s_per_step=(pair[1] / 2000 if valid else ''), pair_max_s_per_step=(pair[2] / 2000 if valid else ''),
                   kspace_avg_s_per_step=(kspace[1] / 2000 if valid else ''), kspace_max_s_per_step=(kspace[2] / 2000 if valid else ''),
                   force_avg_s_per_step=((pair[1] + kspace[1]) / 2000 if valid else ''),
                   force_max_s_per_step=((pair[2] + kspace[2]) / 2000 if valid else ''))
        rows.append(row)

fields = list(rows[0])
with (root / 'results.tsv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter='	')
    writer.writeheader(); writer.writerows(rows)
print(f"complete={sum(r['status']=='complete' for r in rows)}/{len(rows)}")
