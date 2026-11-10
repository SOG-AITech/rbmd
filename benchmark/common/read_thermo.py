"""Read numeric LAMMPS custom thermo blocks, rejecting missing/ambiguous timesteps."""
import math

def read(path):
    columns=None; rows={}
    for line in path.read_text().splitlines():
        fields=line.split()
        if fields and fields[0]=='Step': columns=fields; continue
        if not columns or len(fields)!=len(columns): continue
        try: values=list(map(float,fields))
        except ValueError: continue
        if not all(math.isfinite(v) for v in values): raise ValueError('Non-finite thermo values')
        row=dict(zip(columns,values)); step=int(row['Step'])
        if step in rows and row!=rows[step]: raise ValueError('Different thermo rows at the same step; use one production segment')
        rows[step]=row
    if not rows: raise ValueError(f'No thermo rows in {path}')
    return [rows[k] for k in sorted(rows)]
