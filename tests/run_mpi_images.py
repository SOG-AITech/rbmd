#!/usr/bin/env python3
"""NVT regression for image/unwrapped state through migration and compaction.

Run on allocated GPUs with the normal MPI/toolchain environment. Fixtures have
zero nonbonded forces; full-style trimers additionally exercise bond/angle topology.
The oracle unwraps wrapped coordinates by ID, independently of RBMD's xu fields.
"""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('merge', ROOT / 'scripts/merge_trajectory_shards.py')
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)
LENGTH = 16.0
DT = 0.02
STEPS = 500
FIELDS = ['id', 'type', 'q', 'x', 'y', 'z', 'xu', 'yu', 'zu', 'vx', 'vy', 'vz']


def fixture(style, direction, path, shake=False):
    atoms, velocities, initial, bonds, angles = [], [], [], [], []
    for mol, (i, j, k) in enumerate(itertools.product(range(4), repeat=3), 1):
        center = np.array([0.23 + 4*i, 0.31 + 4*j, 0.47 + 4*k])
        images = np.array([i % 3 - 1, j % 3 - 1, k % 3 - 1])
        offsets = [(0, 0, 0), (.6, 0, 0), (0, .6, 0)] if style == 'full' else [(0, 0, 0)]
        if shake:
            theta = np.deg2rad(109.47)
            offsets = [(0, 0, 0), (1, 0, 0), (float(np.cos(theta)), float(np.sin(theta)), 0)]
        first = len(atoms) + 1
        for offset in offsets:
            atom_id = len(atoms) + 1
            xyz = center + offset
            prefix = f'{atom_id} {mol} 1 0' if style == 'full' else f'{atom_id} 1' + (' 0' if style == 'charge' else '')
            atoms.append(prefix + ' ' + ' '.join(map(str, xyz)) + ' ' + ' '.join(map(str, images)))
            velocities.append(f'{atom_id} {2*direction} {-3*direction} {4*direction}')
            initial.append(xyz + images * LENGTH)
        if style == 'full':
            bonds.extend([f'{len(bonds)+1} 1 {first} {first+1}', f'{len(bonds)+2} 1 {first} {first+2}'])
            angles.append(f'{len(angles)+1} 1 {first+1} {first} {first+2}')
    text = f'MPI image migration regression\n\n{len(atoms)} atoms\n1 atom types\n'
    if style == 'full':
        text += f'{len(bonds)} bonds\n{len(angles)} angles\n0 dihedrals\n0 impropers\n1 bond types\n1 angle types\n'
    text += '\n0 16 xlo xhi\n0 16 ylo yhi\n0 16 zlo zhi\n\nMasses\n\n1 1\n\nPair Coeffs\n\n1 0 1\n'
    if style == 'full':
        bond, angle = (1, 109.47) if shake else (.6, 90)
        text += f'\nBond Coeffs\n\n1 10 {bond}\n\nAngle Coeffs\n\n1 10 {angle}\n'
    text += f'\nAtoms # {style}\n\n' + '\n'.join(atoms) + '\n\nVelocities\n\n' + '\n'.join(velocities) + '\n'
    if style == 'full':
        text += '\nBonds\n\n' + '\n'.join(bonds) + '\n\nAngles\n\n' + '\n'.join(angles) + '\n'
    path.write_text(text)
    return np.array(initial)


def lammps_msd(path, trajectory, reference, binary):
    """Read native RBMD coordinates/images in LAMMPS; never integrate forces."""
    with trajectory.open() as source, (path / 'lammps_native.trj').open('w') as out:
        while (frame := merge.read_frame(source)) is not None:
            step, box, _, fields, rows = frame
            out.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n')
            out.write(''.join(box))
            out.write('ITEM: ATOMS id type x y z ix iy iz\n')
            for row in rows:
                values = row.split()
                xyz = np.array([float(values[fields.index(c)]) for c in ('x', 'y', 'z')])
                xu = np.array([float(values[fields.index(c)]) for c in ('xu', 'yu', 'zu')])
                images = np.rint((xu - xyz) / LENGTH).astype(int)
                assert np.max(abs(xyz + images * LENGTH - xu)) < 5e-4
                out.write(values[fields.index('id')] + ' 1 ' + ' '.join(map(str, xyz)) +
                          ' ' + ' '.join(map(str, images)) + '\n')
    (path / 'in.msd').write_text("""units lj
atom_style atomic
boundary p p p
region box block 0 16 0 16 0 16
create_box 1 box
read_dump lammps_native.trj 0 x y z ix iy iz box yes add keep
mass 1 1
pair_style zero 2.5
pair_coeff * *
timestep 0.02
compute displacement all msd com no average no
fix result all ave/time 5 1 5 c_displacement[4] file lammps_msd.txt
thermo 500
thermo_style custom step atoms c_displacement[4]
rerun lammps_native.trj first 0 last 500 every 5 dump x y z ix iy iz box yes
""")
    with (path / 'lammps.log').open('w') as log:
        run = subprocess.run(['mpirun', '-np', '1', str(binary), '-in', 'in.msd'], cwd=path,
                             stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert run.returncode == 0, (path, 'LAMMPS rerun', run.returncode)
    data = np.loadtxt(path / 'lammps_msd.txt')
    assert set(data[:, 0].astype(int)) == set(range(0, STEPS+1, 5)), path
    error = 0.0
    for step, msd in data:
        expected = reference[int(step)]
        error = max(error, abs(msd - expected))
        assert np.isclose(msd, expected, atol=3e-4, rtol=2e-5), (path, 'LAMMPS MSD mismatch', step, msd, expected)
    return error


def verify(path, initial, ranks, direction, selected, lammps=None):
    shards = sorted(path.glob('rbmd.rank*.trj'))
    if shards:
        assert len(shards) == ranks, (path, len(shards), ranks)
        merge.merge(shards, path / 'merged.trj')
        trajectory = path / 'merged.trj'
    else:
        trajectory = path / 'rbmd.trj'
    previous = None
    unwrapped = initial.copy()
    max_error = 0.0
    crossings = np.zeros(3, dtype=int)
    max_jump = 0.0
    max_msd_error = 0.0
    reference_msds = {}
    count = 0
    with trajectory.open() as f:
        while (frame := merge.read_frame(f)) is not None:
            step, box, _, fields, rows = frame
            assert step == count * 5, (path, step, count)
            a = np.array([[float(x) for x in r.split()] for r in rows])
            a = a[np.argsort(a[:, fields.index('id')])]
            assert np.array_equal(a[:, fields.index('id')], np.arange(1, len(initial)+1)), path
            assert np.isfinite(a).all(), path
            xyz = a[:, [fields.index(c) for c in ('x', 'y', 'z')]]
            xu = a[:, [fields.index(c) for c in ('xu', 'yu', 'zu')]]
            assert np.all((xyz >= -1e-5) & (xyz < LENGTH + 1e-5)), path
            if previous is not None:
                delta = xyz - previous
                jumps = np.rint(delta / LENGTH)
                crossings += np.sum(np.abs(jumps).astype(int), axis=0)
                delta -= jumps * LENGTH
                max_jump = max(max_jump, float(np.max(np.abs(delta))))
                assert np.max(np.abs(delta)) < 1.0, (path, 'sampling too sparse')
                unwrapped += delta
            previous = xyz
            error = float(np.max(abs(xu - unwrapped)))
            max_error = max(max_error, error)
            assert error < 5e-4, (path, 'unwrapped mismatch', step, error)
            reference_msd = float(np.mean(np.sum((unwrapped - initial)**2, axis=1)))
            reference_msds[step] = reference_msd
            native_msd = float(np.mean(np.sum((xu - initial)**2, axis=1)))
            max_msd_error = max(max_msd_error, abs(native_msd - reference_msd))
            assert np.isclose(native_msd, reference_msd, atol=3e-4, rtol=2e-5), (path, "MSD mismatch", step)
            count += 1
    assert count == STEPS // 5 + 1, (path, count)
    assert np.all(crossings > 0), (path, crossings)
    # Travel farther than a full box in every dimension: includes migration after wrapping.
    displacement = unwrapped - initial
    assert np.all(np.min(np.abs(displacement), axis=0) > LENGTH), (path, displacement)
    assert np.all(np.sign(displacement) == np.sign(np.array([2, -3, 4]) * direction)), path
    migration_sends = 0
    for csv_path in path.glob('**/*exchange_stage_stats*.csv'):
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row['stage'] == 'after_leaving':
                    migration_sends += int(row['send_atoms'])
    owner_changes = 0
    if selected:
        owners_by_step = {}
        for shard in shards:
            with shard.open() as f:
                while (frame := merge.read_frame(f)) is not None:
                    owners = owners_by_step.setdefault(frame[0], {})
                    for row in frame[4]:
                        atom_id = int(row.split()[0])
                        assert atom_id not in owners
                        owners[atom_id] = shard.name
        previous_owners = None
        for step, owners in sorted(owners_by_step.items()):
            if previous_owners is not None:
                owner_changes += sum(previous_owners[i] != rank for i, rank in owners.items())
            previous_owners = owners
    if ranks > 1:
        assert migration_sends > 0, (path, 'no MPI migration exercised')
        if selected:
            assert owner_changes > 0, (path, 'no ownership changes')
    lammps_error = lammps_msd(path, trajectory, reference_msds, lammps) if lammps else None
    return dict(lammps_msd_max_error=lammps_error, atoms=len(initial), frames=count, max_unwrapped_error=max_error,
                periodic_crossings=crossings.tolist(), max_frame_jump=max_jump,
                migration_sends=migration_sends, owner_changes=owner_changes,
                max_msd_error=max_msd_error, native_msd_verified=True, initial_nonzero_images=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=[1, 2, 8], required=True)
    parser.add_argument('--lammps', type=Path)
    parser.add_argument('--case', choices=['atomic', 'charge', 'full', 'full_shake'])
    args = parser.parse_args()
    binary = args.binary.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for case in ([args.case] if args.case else ['atomic', 'charge', 'full', 'full_shake']):
        shake = case == 'full_shake'
        style = 'full' if shake else case
        for direction in (1, -1):
            path = root / f'{case}_{"positive" if direction == 1 else "negative"}'
            path.mkdir()
            initial = fixture(style, direction, path / 'atoms.data', shake)
            force = {'type': {'atomic': 'LJ/CUT', 'charge': 'LJ/CUT/COUL/LONG', 'full': 'CVFF'}[style]}
            if style == 'full':
                force.update(bond_type='harmonic', angle_type='harmonic', dihedral_type='harmonic')
            n = len(initial)
            temperature = 29*n/(3*n-3-(n if shake else 0))
            config = {
                'init_configuration': {'read_data': {'file': str(path/'atoms.data'), 'unit': 'LJ',
                    'atom_style': style, 'velocity_type': 'NONE', 'replicate': [1, 1, 1]}},
                'hyper_parameters': {'force_field': force, 'neighbor': {'type': 'Verlet-List',
                    'cut_off': 2.5, 'r_core': 1.25, 'neighbor_sample_num': 16, 'energy_rbl_flag': 'no'},
                    'extend': {'fix_shake': shake, 'special_bonds': [0, 0, 0]}},
                'execution': {'ensemble': 'NVT', 'integration_type': 'vv', 'group': 'all', 'com_bias': 'no',
                    'temp_ctrl_type': 'NOSE_HOOVER', 'press_ctrl_type': 'NOSE_HOOVER',
                    'temperature': [temperature, temperature, 1000], 'par_a': 1, 'par_b': 3,
                    'timestep': DT, 'num_steps': STEPS},
                'outputs': {'thermo_out': {'interval': 100}, 'trajectory_out': {'interval': 5}}}
            if style != 'atomic':
                config['hyper_parameters']['coulomb'] = {'type': 'EWALD', 'debug_alpha': True, 'alpha': .1,
                    'kmax': [2, 2, 2], 'coulomb_sample_num': 16, 'energy_rbe_flag': 'no'}
            if direction < 0:
                config['outputs']['trajectory_out']['fields'] = FIELDS
            (path/'run.json').write_text(json.dumps(config, indent=2)+'\n')
            env = dict(os.environ, RBMD_DEBUG_EXCHANGE_STAGE_STATS='1')
            with (path/'run.log').open('w') as log:
                run = subprocess.run(['mpirun', '-np', str(args.ranks), str(binary), '-j', str(path/'run.json')],
                                     cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            assert run.returncode == 0, (path, run.returncode)
            result = verify(path, initial, args.ranks, direction, direction < 0, args.lammps.resolve() if args.lammps else None)
            result.update(style=style, shake=shake, direction=direction, ranks=args.ranks)
            results.append(result)
            print('PASS ' + json.dumps(result), flush=True)
    (root/'PASS.json').write_text(json.dumps({'binary': str(binary),
        'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'results': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
