#!/usr/bin/env python3
"""G06 GPU smoke: default RBE, replicated/explicit geometry, LJ/water/PEO.

Use an allocated compute node. --ranks 0 executes a non-MPI binary directly.
These short NVT/force-only checks are not accuracy or equilibration evidence.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def ions(path, dims):
    atoms = []
    for a, b, c in itertools.product(*(range(v) for v in dims)):
        for x, y, z in itertools.product(range(4), repeat=3):
            q = 1 if (x+y+z) % 2 else -1
            atoms.append(f'{len(atoms)+1} {1 if q > 0 else 2} {q} '
                         f'{2+4*x+.07*y+16*a} {2+5*y+.03*z+20*b} {2+6*z+.05*x+24*c} 0 0 0')
    path.write_text(f'G06 ions\n\n{len(atoms)} atoms\n2 atom types\n\n'
                    f'0 {16*dims[0]} xlo xhi\n0 {20*dims[1]} ylo yhi\n0 {24*dims[2]} zlo zhi\n'
                    '\nMasses\n\n1 1\n2 1\n\nPair Coeffs\n\n1 1 1\n2 1 1\n\nAtoms # charge\n\n'
                    + '\n'.join(atoms) + '\n')


def water(path):
    atoms, bonds, angles = [], [], []
    for mol, (x, y, z) in enumerate(itertools.product(range(3), repeat=3), 1):
        first = len(atoms)+1
        for typ, q, dx, dy in [(1, -.8476, 0, 0), (2, .4238, 1, 0),
                               (2, .4238, math.cos(math.radians(109.47)), math.sin(math.radians(109.47)))]:
            atoms.append(f'{len(atoms)+1} {mol} {typ} {q} {8+14*x+dx} {8+14*y+dy} {8+14*z} 0 0 0')
        bonds.extend([f'{len(bonds)+1} 1 {first} {first+1}', f'{len(bonds)+2} 1 {first} {first+2}'])
        angles.append(f'{len(angles)+1} 1 {first+1} {first} {first+2}')
    path.write_text('G06 dilute SPC/E smoke fixture\n\n81 atoms\n2 atom types\n54 bonds\n1 bond types\n'
                    '27 angles\n1 angle types\n0 dihedrals\n0 impropers\n\n0 48 xlo xhi\n0 48 ylo yhi\n0 48 zlo zhi\n'
                    '\nMasses\n\n1 15.9994\n2 1.00794\n\nPair Coeffs\n\n1 .1556 3.166\n2 0 0\n'
                    '\nBond Coeffs\n\n1 1000 1\n\nAngle Coeffs\n\n1 100 109.47\n'
                    '\nAtoms # full\n\n'+'\n'.join(atoms)+'\n\nBonds\n\n'+'\n'.join(bonds)+
                    '\n\nAngles\n\n'+'\n'.join(angles)+'\n')


def read_forces(path):
    values = {}
    for file in path.glob('revision.forces*.csv'):
        with file.open() as f:
            for row in csv.DictReader(f):
                key = (int(row['sample']), int(row['id']))
                assert key not in values, (path, key)
                values[key] = tuple(float(row[k]) for k in ('x', 'y', 'z', 'fx', 'fy', 'fz'))
    assert values and all(math.isfinite(v) for row in values.values() for v in row), path
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    binary, root = args.binary.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    ions(root/'ions.data', [1, 1, 1]); ions(root/'expanded.data', [2, 3, 1]); water(root/'water.data')
    peo = ROOT/'examples/01-peo/peo+Li.data'
    n_peo = int(re.search(r'^(\d+) atoms', peo.read_text(), re.M).group(1))
    base = {
        'init_configuration': {'read_data': {'unit': 'LJ', 'atom_style': 'charge', 'velocity_type': 'GAUSS'}},
        'hyper_parameters': {'force_field': {'type': 'LJ/CUT/COUL/LONG'},
            'neighbor': {'type': 'Verlet-List', 'cut_off': 3, 'r_core': 1.5, 'neighbor_sample_num': 16, 'energy_rbl_flag': 'no'},
            'coulomb': {'type': 'RBE', 'accuracy': 1e-4, 'coulomb_sample_num': 64,
                        'energy_rbe_flag': 'no', 'random_seed': 20261016},
            'extend': {'fix_shake': False, 'special_bonds': [0, 0, 0]}},
        'execution': {'ensemble': 'NVT', 'group': 'all', 'com_bias': 'no', 'integration_type': 'vv',
            'par_a': 1, 'par_b': 3, 'temperature': [1, 1, 1], 'temp_ctrl_type': 'NOSE_HOOVER',
            'press_ctrl_type': 'NOSE_HOOVER', 'timestep': .001, 'num_steps': 0,
            'diagnostics': {'fixed_samples': 3, 'forces': True, 'energy': True, 'interval': 1, 'prefix': 'revision'}},
        'outputs': {'thermo_out': {'interval': 1}},
    }
    cases = [('lj_base', root/'ions.data', None, 64, False),
             ('lj_replicate', root/'ions.data', 'replicate', 384, False),
             ('lj_alias', root/'ions.data', 'replicate_dims', 384, False),
             ('lj_expanded', root/'expanded.data', None, 384, False),
             ('lj_nvt', root/'ions.data', 'replicate', 384, True),
             ('lj_rbl_nvt', root/'ions.data', 'replicate', 384, True)]
    for system, data, n in [('water', root/'water.data', 81), ('peo', peo, n_peo)]:
        cases.extend([(system+'_base', data, None, n, False),
                      (system+'_replicate', data, 'replicate', 2*n, False),
                      (system+'_nvt', data, 'replicate', 2*n, True),
                      (system+'_rbl_nvt', data, 'replicate', 2*n, True)])
    results, lj_forces = [], {}
    for name, data, replicate_key, n, dynamics in cases:
        path = root/name; path.mkdir()
        cfg = copy.deepcopy(base)
        rd = cfg['init_configuration']['read_data']; rd['file'] = str(data)
        if replicate_key:
            rd[replicate_key] = [2, 3, 1] if name.startswith('lj') else [2, 1, 1]
        if not name.startswith('lj'):
            rd.update(unit='REAL', atom_style='full')
            cfg['hyper_parameters']['neighbor'].update(cut_off=9, r_core=5)
            cfg['hyper_parameters']['force_field'] = dict(type='CVFF', bond_type='harmonic', angle_type='harmonic', dihedral_type='harmonic')
            cfg['hyper_parameters']['extend']['fix_shake'] = name.startswith('water')
            cfg['execution'].update(temperature=[298, 298, 100], timestep=.01)
        if '_rbl_' in name:
            cfg['hyper_parameters']['neighbor'].update(type='RBL', energy_rbl_flag='yes')
        if dynamics:
            cfg['execution']['diagnostics'].pop('fixed_samples')
            cfg['execution']['num_steps'] = 3
        (path/'run.json').write_text(json.dumps(cfg, indent=2)+'\n')
        command = ([str(binary)] if args.ranks == 0 else ['mpirun', '-np', str(args.ranks), str(binary)])
        with (path/'run.log').open('w') as log:
            run = subprocess.run(command+['-j', str(path/'run.json')], cwd=path, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=240)
        assert run.returncode == 0, (path, run.returncode)
        log = (path/'run.log').read_text(errors='replace')
        assert 'rbe_sampling : full simulation box' in log, path
        values = read_forces(path)
        steps = {s for s, _ in values}
        assert {1, 2, 3}.issubset(steps) and len(values) == n*len(steps), (path, len(values), n, steps)
        assert all(sum(s == step for s, _ in values) == n for step in steps), path
        with (path/'revision.observables.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert rows and all(math.isfinite(float(row[k])) for row in rows
                            for k in ('step', 'time', 'n_atoms', 'kinetic_total', 'potential_total', 'total_energy')), path
        if name in ('lj_replicate', 'lj_alias', 'lj_expanded'):
            # Reader ID order can differ; compare the identical frozen physical coordinates.
            lj_forces[name] = {(s,)+tuple(round(v, 4) for v in row[:3]): row[3:] for (s, _), row in values.items()}
            assert len(lj_forces[name]) == len(values)
        result = dict(case=name, atoms=n, samples=sorted(steps), ranks=args.ranks, passed=True)
        results.append(result); print('PASS '+json.dumps(result), flush=True)
    if args.ranks:
        reference = lj_forces['lj_expanded']
        for name in ('lj_replicate', 'lj_alias'):
            values = lj_forces[name]
            assert values.keys() == reference.keys(), name
            assert all(math.isclose(a, b, rel_tol=3e-4, abs_tol=3e-5)
                       for key in values for a, b in zip(values[key], reference[key])), name
        print('PASS identical geometry: replicate/alias/explicit full box', flush=True)
    (root/'PASS.json').write_text(json.dumps(dict(binary=str(binary),
        sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), cases=results,
        replicated_geometry_force_comparison=bool(args.ranks)), indent=2)+'\n')


if __name__ == '__main__':
    main()
