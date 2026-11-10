#!/usr/bin/env python3
"""Reduce LAMMPS observables, retaining image/convergence failures as explicit status."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from read_thermo import read


def fit(x, y):
    mx, my = st.mean(x), st.mean(y)
    slope = sum((a-mx)*(b-my) for a, b in zip(x, y)) / sum((a-mx)**2 for a in x)
    intercept = my-slope*mx
    residual = sum((b-slope*a-intercept)**2 for a, b in zip(x, y))
    total = sum((b-my)**2 for b in y)
    return dict(slope=slope, intercept=intercept, r_squared=1-residual/total if total else None)


def epsilon(rows, temperature):
    variance = sum(st.pvariance([r[f'c_dip_local[{i}]'] for r in rows]) for i in (1, 2, 3))
    return 1 + variance * (1.602176634e-29)**2 / (
        3 * 8.8541878128e-12 * rows[0]['Volume'] * 1e-30 * 1.380649e-23 * temperature)


def rdf_average(path, first, last, stride):
    lines = iter(l for l in path.read_text().splitlines() if l.strip() and not l.startswith('#'))
    steps, frames, radii = [], [], None
    for line in lines:
        step, count = map(int, line.split())
        data = [list(map(float, next(lines).split())) for _ in range(count)]
        assert all(len(r) == 8 and r[0] == i+1 and all(map(math.isfinite, r)) for i, r in enumerate(data))
        if not first <= step <= last:
            continue
        grid = [r[1] for r in data]
        if radii is not None:
            assert grid == radii
        radii = grid
        frames.append([r[2::2] for r in data])
        steps.append(step)
    assert steps == list(range(first, last+1, stride)), 'Incomplete RDF output'
    return radii, [[st.mean(f[i][j] for f in frames) for j in range(3)] for i in range(len(radii))], len(steps)


def reduce(folder, smoke=False):
    meta = json.loads((folder / 'postprocess.json').read_text())
    rows = read(folder / 'log.lammps')
    assert [r['Step'] for r in rows] == list(range(0, meta['last']+1, meta['stride']))
    assert all(r['Atoms'] == meta['n_atoms'] for r in rows)
    assert all(r['Volume'] == rows[0]['Volume'] for r in rows)
    assert abs(rows[0]['c_msd1[4]']) < 1e-10, 'MSD reference is not the first snapshot'
    water = meta['system'] == 'water'
    geometry = None
    if water:
        assert all(abs(r['c_qsum']) < 1e-6 and r['c_omin'] == r['c_omax'] == 1 for r in rows)
        geometry = dict(min_OH=min(r['c_ohmin'] for r in rows), max_OH=max(r['c_ohmax'] for r in rows))
        assert 0.95 < geometry['min_OH'] <= geometry['max_OH'] < 1.05, 'Molecular geometry invalid'
        if meta['source_engine'] == 'lammps':
            delta = max(abs(r[f'c_dip_local[{i}]']-r[f'c_dip_raw[{i}]']) for r in rows for i in (1, 2, 3))
            geometry['max_local_vs_native_dipole_difference_eA'] = delta
            assert delta < 1e-6, 'Molecular dipole failed comparison with native compute dipole'
    else:
        assert all(r['c_qnorm'] == rows[0]['c_qnorm'] > 0 for r in rows)
        assert abs(rows[0]['c_msd2[4]']) < 1e-10
    if smoke:
        # ave/time may omit step zero, depending on setup; later production windows do not.
        rdf_average(folder / 'rdf.dat', 100, meta['last'], 100)
        result = dict(status='PASS', frames=len(rows), geometry=geometry)
        (folder / 'smoke.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    first, last = 20000, meta['last']
    assert last == 100000, 'This round-one reduction requires the complete 100000-step run'
    selected = [r for r in rows if first <= r['Step'] <= last]
    case = Path(meta['source_case'])
    check_path = case.parents[3] / 'checks-round1' / f"{meta['system']}-{meta['source_engine']}.json"
    check = json.loads(check_path.read_text())
    assert check['frames'] == 1001 and check['finite_values'] and check['complete_unique_ids']
    box_min = min(rows[0][f'L{a}'] for a in 'xyz')
    images_valid = check['max_displacement_between_frames'] < box_min / 2
    if water:
        images_valid = images_valid and check['molecular_images_consistent']
    result = dict(
        status='EXPLORATORY_ONE_ROUND', system=meta['system'], engine=meta['source_engine'],
        analysis_steps=[first, last], sampled_frames=len(selected), geometry=geometry,
        uncertainty='Only round 1 of 3. No independent-repeat error bar or convergence certification.',
        analysis_window_note='Discard first 20%; this is a declared window, not proof of equilibration.',
        source_image_check=check, diffusion={}, independent_repeats_completed=1,
    )
    groups = ['1'] if water else ['1', '2']
    fitrows = [r for r in rows if 50000 <= r['Step'] <= last]
    for group in groups:
        fits = {}
        for low, high in [(50000, 100000), (50000, 75000), (75000, 100000)]:
            part = [r for r in fitrows if low <= r['Step'] <= high]
            f = fit([r['Step']*meta['timestep'] for r in part], [r[f'c_msd{group}[4]'] for r in part])
            fits[f'{low}:{high}'] = f
        f = fits['50000:100000']
        result['diffusion'][group] = dict(
            status='ESTIMATE_NOT_CONVERGENCE_CERTIFIED' if images_valid else 'INVALID_UNWRAPPED_IMAGES',
            D=f['slope']/6*(1e-5 if water else 1) if images_valid else None,
            units='m^2/s' if water else 'sigma^2/tau',
            raw_msd_fit_diagnostic=fits, group='oxygen' if water else f'ion_type_{group}',
            com_removed=True, msd_origin_step=0, fit_steps=[50000, 100000],
        )
    radii, curves, frames = rdf_average(folder / 'rdf.dat', first, last, meta['stride'])
    with (folder / 'rdf_mean.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['r', 'g11', 'g12', 'g22'])
        writer.writerows([r, *g] for r, g in zip(radii, curves))
    result['rdf'] = dict(file='rdf_mean.csv', frames=frames, dr=meta['dr'], radius=meta['radius'],
                         exclusions=meta['rdf_exclusions'])
    if water:
        if meta['source_engine'] == 'rbmd':
            temperatures = {}
            for line in (case / 'temperature.txt').read_text().splitlines():
                try:
                    step, temp = map(float, line.split()[:2])
                except ValueError:
                    continue
                if first <= step <= last:
                    temperatures[int(step)] = temp
        else:
            temperatures = {int(r['Step']): r['Temp'] for r in read(case / 'log.lammps') if first <= r['Step'] <= last}
        assert sorted(temperatures) == list(range(first, last+1, meta['stride']))
        assert all(math.isfinite(t) and t > 0 for t in temperatures.values())
        temperature = st.mean(temperatures.values())
        blocks = []
        for low in range(first, last, 20000):
            part = [r for r in selected if low <= r['Step'] < low+20000]
            blocks.append(dict(first=low, last=low+19900,
                               epsilon_r=epsilon(part, st.mean(temperatures[int(r['Step'])] for r in part))))
        result['dielectric'] = dict(epsilon_r=epsilon(selected, temperature), temperature_K=temperature,
                                    temperature_basis='original trajectory, same 801 saved timesteps',
                                    boundary='conducting, fixed volume, neutral molecules',
                                    method=meta['dipole_method'], block_estimates=blocks,
                                    block_note='Descriptive block estimates; not independent-repeat confidence intervals.')
    else:
        result['Szz'] = {
            f'{axis}{mode}': dict(k=2*math.pi*mode/rows[0]['L'+axis],
                                  mean=st.mean(r[f'v_S{axis}{mode}'] for r in selected))
            for mode in range(1, 5) for axis in 'xyz'
        }
        result['Szz_normalization'] = '|sum(q exp(ikr))|^2 / sum(q^2); axial k; no axis pooling'
    with (folder / 'observables.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    result = reduce(args.folder, args.smoke)
    print(json.dumps({'folder': str(args.folder), 'status': result['status']}))
