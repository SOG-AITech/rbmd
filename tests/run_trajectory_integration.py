#!/usr/bin/env python3
"""Run actual GPU/MPI executable and compare local selection with full frames.

Run in a fresh working directory. RBMD_BIN and RBMD_SOURCE locate the build.
MPI ranks default to 4; environment/device binding comes from the caller.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(os.environ.get('RBMD_SOURCE', Path(__file__).resolve().parents[1]))
BIN = os.environ['RBMD_BIN']
RANKS = int(os.environ.get('TRAJECTORY_TEST_RANKS', '4'))
spec = importlib.util.spec_from_file_location('merge', ROOT/'scripts/merge_trajectory_shards.py')
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)
base = json.loads((ROOT/'examples/03-metal/au-sigma5-grain-boundary/Au.sigma5-gb.local-trajectory.json').read_text())
base['init_configuration']['read_data']['file'] = str(ROOT/'examples/03-metal/au-sigma5-grain-boundary/Au.sigma5-gb.data')
base['hyper_parameters']['force_field']['potential_file'] = str(ROOT/'examples/03-metal/Au/Au_u3.eam')
base['hyper_parameters']['neighbor'].update(type='VERLET-LIST', direct_traversal=False)
base['execution']['num_steps'] = 20
base['outputs']['trajectory_out'].pop('selection')
base['outputs']['trajectory_out']['interval'] = 10
FIELDS = base['outputs']['trajectory_out']['fields']


def run(name, selection=None, cap=None, expect_fail=False, ranks=RANKS):
    path = Path(name).resolve()
    path.mkdir()
    config = copy.deepcopy(base)
    out = config['outputs']['trajectory_out']
    if selection is not None:
        out['selection'] = selection
    if cap is not None:
        out['max_atoms'] = cap
    (path/'run.json').write_text(json.dumps(config, indent=2))
    with (path/'run.log').open('w') as log:
        result = subprocess.run(['mpirun', '-np', str(ranks), BIN, '-j', str(path/'run.json')],
                                cwd=path, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    if expect_fail:
        assert result.returncode != 0
        assert 'exceeds global max_atoms' in (path/'run.log').read_text()
        return
    assert result.returncode == 0, (name, result.returncode)
    assert 'Finish' in (path/'run.log').read_text()
    shards = sorted(path.glob('rbmd.rank*.trj'))
    assert len(shards) == ranks
    assert merge.merge(shards, path/'merged.trj') == 3
    frames = []
    with (path/'merged.trj').open() as f:
        while True:
            frame = merge.read_frame(f)
            if frame is None:
                break
            assert frame[3] == FIELDS
            rows = {int(r.split()[0]): r.split() for r in frame[4]}
            assert all(len(row) == len(FIELDS) for row in rows.values())
            frames.append((frame[0], rows))
    assert [s for s, _ in frames] == [0, 10, 20]
    return frames


full = run('full')
assert all(len(rows) == 2528 for _, rows in full)
region = {'x': [0, 13], 'y': [-10, 10]}

def inside(row):
    return 0 <= float(row[2]) < 13 and -10 <= float(row[3]) < 10

expected = [{i for i, row in rows.items() if inside(row)} for _, rows in full]
assert 0 < len(expected[0]) < 2528
for mode in ('dynamic', 'initial'):
    frames = run(mode, {'mode': mode, 'region': region})
    for k, (_, rows) in enumerate(frames):
        assert set(rows) == expected[k if mode == 'dynamic' else 0], (mode, k)
        # Identical simulation, with only its observation configuration changed.
        for i, row in rows.items():
            assert row[:2] == full[k][1][i][:2]
            assert all(abs(float(a)-float(b)) < 1e-4 for a,b in zip(row[2:], full[k][1][i][2:]))
# Verify this fixture really exercises region exit and MPI ownership changes.
with Path('initial/merged.trj').open() as f:
    fixed_frames = [merge.read_frame(f) for _ in range(3)]
assert any(not inside(row.split()) for row in fixed_frames[-1][4])
if RANKS > 1:
    owners = [{} for _ in range(3)]
    for shard in Path('initial').glob('rbmd.rank*.trj'):
        with shard.open() as f:
            for k in range(3):
                for row in merge.read_frame(f)[4]:
                    owners[k][int(row.split()[0])] = shard.name
    assert any(owners[0][i] != owners[-1][i] for i in owners[0])
frames = run('types-regions', {'types':[2], 'regions':[region, {'x':[20,25]}]})
for k, (_, rows) in enumerate(frames):
    wanted={i for i,r in full[k][1].items() if int(r[1])==2 and (inside(r) or 20 <= float(r[2]) < 25)}
    assert set(rows)==wanted
frames = run('ids', {'ids':[1,2528]})
assert all(set(rows)=={1,2528} for _,rows in frames)
frames = run('empty', {'mode':'initial', 'region':{'x':[1000,1001]}})
assert all(not rows for _,rows in frames)
run('cap', {'types':[1,2]}, cap=10, expect_fail=True)
frames = run('single-rank', {'ids':[1,2528]}, ranks=1)
assert all(set(rows)=={1,2528} for _,rows in frames)
Path('PASS.json').write_text(json.dumps({'ranks':RANKS,'cases':8,'steps':[0,10,20],
    'initial_atoms':len(expected[0]),'status':'passed'},indent=2))
print('GPU/MPI trajectory integration tests passed', flush=True)
