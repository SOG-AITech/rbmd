#!/usr/bin/env python3
"""Prepare bounded-memory LAMMPS replay of the original serial or parallel dumps."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(case, output, last):
    cfg = json.loads((case / 'rbmd.json').read_text())
    receipt = json.loads((case / 'run_receipt.json').read_text())
    preflight = json.loads((case / 'preflight.json').read_text())
    assert receipt['exit_code'] == 0 and preflight['status'] == 'PASS'
    data = cfg['init_configuration']['read_data']
    water = data['unit'].lower() == 'real'
    source = case / 'rbmd.rank%.trj' if (case / 'rbmd.rank0.trj').exists() else case / 'trajectory.dump'
    parallel = ' nfile 4' if '%' in str(source) else ''
    if parallel:
        assert receipt['ranks'] == 4
    assert last > 0 and last % 100 == 0
    # Explicit xu labels reset images on every read; wrapped coordinates alone lose crossings.
    read = 'x y z vx vy vz box yes label x xu label y yu label z zu scaled no wrapped no' + parallel
    radius = 9 if water else 16
    lines = [
        f"units {data['unit'].lower()}", f"atom_style {data['atom_style']}",
        'boundary p p p',
    ]
    if water:
        lines += ['bond_style zero', 'angle_style zero']
    lines += [
        f'pair_style zero {radius}', f'read_data "{data["file"]}" nocoeff',
        'pair_coeff * *',
    ]
    if water:
        lines += ['bond_coeff *', 'angle_coeff *']
    lines += [
        'special_bonds lj/coul 0 0 0',
        'neighbor 1 bin', 'neigh_modify every 1 delay 0 check no',
        f'comm_modify cutoff {radius+1}',
        f"timestep {cfg['execution']['timestep']}",
        f'read_dump "{source}" 0 {read}',
        f'compute rdf_all all rdf {radius*20} 1 1 1 2 2 2 cutoff {radius}',
        'fix rdf_output all ave/time 100 1 100 c_rdf_all[*] file rdf.dat mode vector',
        'group species1 type 1', 'group species2 type 2',
        'compute msd1 species1 msd com yes average no',
    ]
    fields = 'step atoms vol lx ly lz c_msd1[1] c_msd1[2] c_msd1[3] c_msd1[4]'
    if water:
        # A neutral, compact molecule's dipole is origin-independent. Anchor H on its
        # own O using the unique short O-H bond, not temporal minimum-image guessing.
        lines += [
            'compute mol all chunk/atom molecule nchunk once ids once compress yes',
            'variable ox atom (type==1)*x', 'variable oy atom (type==1)*y',
            'variable oz atom (type==1)*z', 'variable isO atom (type==1)',
            'compute anchor all reduce/chunk mol sum v_ox v_oy v_oz v_isO',
            'compute spread all chunk/spread/atom mol c_anchor[*]',
        ]
        for i, axis in enumerate('xyz', 1):
            lines += [
                f'variable d{axis} atom {axis}-c_spread[{i}]-l{axis}*floor(({axis}-c_spread[{i}])/l{axis}+0.5)',
                f'variable m{axis} atom q*v_d{axis}',
            ]
        lines += [
            'compute dip_local all reduce sum v_mx v_my v_mz',
            'compute dip_raw all dipole',
            'variable oh atom sqrt(v_dx*v_dx+v_dy*v_dy+v_dz*v_dz)',
            'compute ohmin species2 reduce min v_oh',
            'compute ohmax species2 reduce max v_oh',
            'compute omin all reduce min c_spread[4]',
            'compute omax all reduce max c_spread[4]',
            'variable charge atom q', 'compute qsum all reduce sum v_charge',
        ]
        fields += ' c_dip_local[1] c_dip_local[2] c_dip_local[3] c_dip_raw[1] c_dip_raw[2] c_dip_raw[3] c_ohmin c_ohmax c_omin c_omax c_qsum'
    else:
        lines += [
            'compute msd2 species2 msd com yes average no',
            'variable q2 atom q*q', 'compute qnorm all reduce sum v_q2',
        ]
        fields += ' c_msd2[1] c_msd2[2] c_msd2[3] c_msd2[4] c_qnorm'
        for mode in range(1, 5):
            for axis in 'xyz':
                tag = f'{axis}{mode}'
                lines += [
                    f'variable re{tag} atom q*cos(2*PI*{mode}*{axis}/l{axis})',
                    f'variable im{tag} atom q*sin(2*PI*{mode}*{axis}/l{axis})',
                    f'compute rho{tag} all reduce sum v_re{tag} v_im{tag}',
                    f'variable S{tag} equal (c_rho{tag}[1]^2+c_rho{tag}[2]^2)/c_qnorm',
                ]
                fields += f' v_S{tag}'
    lines += [
        f'thermo_style custom {fields}',
        'thermo_modify norm no lost error flush yes format float %.17g', 'thermo 100',
        f'rerun "{source}" first 0 last {last} every 100 dump {read}',
    ]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run.in').write_text('\n'.join(lines) + '\n')
    meta = dict(
        source_case=str(case), source=str(source), system='water' if water else 'lj',
        source_engine=case.name, n_atoms=preflight['n_atoms'], last=last, stride=100,
        timestep=cfg['execution']['timestep'], units=data['unit'].lower(),
        radius=radius, dr=0.05, rdf_pairs=[[1, 1], [1, 2], [2, 2]],
        rdf_exclusions='1-2, 1-3 and 1-4 bonded pairs excluded',
        msd_origin_step=0, msd_com_removed=True,
        dipole_method='LAMMPS molecule chunks; neutral OHH; minimum-image H relative to its own O',
        forces='zero; no dynamics or PPPM rerun required for these observables',
        production_binary_sha256=receipt['binary_sha256'],
        source_config_sha256=hashlib.sha256((case / 'rbmd.json').read_bytes()).hexdigest(),
    )
    (output / 'postprocess.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--last', type=int, default=100000)
    args = parser.parse_args()
    prepare(args.case.resolve(), args.output.resolve(), args.last)
