#!/usr/bin/env python3
"""Exercise production molecular LJ kernels on a host (zero-charge pairs).

Checks actual table construction, RBL energy and SOG force/energy, and compiles
GPU op forwarding with launch syntax removed. Does not emulate GPU scheduling.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from run_cvff_bonded_host import extract_definition

ROOT = Path(__file__).resolve().parents[1]


def main():
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        raise SystemExit('A C++17 compiler is required')
    source = (ROOT / 'src/force/src/cvff_op/rocm/cvff_op.hip.cu').read_text()
    lj = (ROOT / 'src/force/src/lj_op/rocm/lj_op.hip.cu').read_text()
    declarations = (ROOT / 'src/force/src/cvff_op/cvff_op.h').read_text()
    # Ignore braces in comments when finding function boundaries.
    strip_comments = lambda text: re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.DOTALL)
    source, lj = strip_comments(source), strip_comments(lj)
    definitions = [extract_definition(lj, 'inline __device__ void ' + name + '(')
                   for name in ('lj126', 'ComputeVirial')]
    for marker in ('inline __device__ void LoadSpecialCounts(',
                   'inline __device__ rbmd::Real LookupSpecialWeight(',
                   'inline __device__ void LJForceEnergyFromTypePair(',
                   '__global__ void ComputeSpecialLJCutCoulEnergy(',
                   '__global__ void ComputeSpecialLJCutCoulForceUserKernel('):
        definitions.append(extract_definition(source, marker))
    for name in ('SpeciaLJCutCoulEnergyOp', 'ComputeSpecialLJCutCoulForceUserOp'):
        declaration = extract_definition(declarations, 'struct ' + name + '<device::DEVICE_GPU>')
        definition = extract_definition(source, 'void ' + name + '<device::DEVICE_GPU>::operator()(')
        definition = re.sub(r'<<<.*?>>>', '', definition, flags=re.DOTALL)
        definitions.append('template<typename> struct ' + name + ';\ntemplate<>\n' + declaration + ';\n' + definition)
    host = (ROOT / 'src/force/src/cvff.cpp').read_text()
    loop = extract_definition(host, 'for (rbmd::Id type_i = 0; type_i < _lj_num_types; ++type_i)')
    definitions.append('void build_table(const rbmd::Real* h_sigma, const rbmd::Real* h_eps, '
                       'bool _geometric_lj_mixing, rbmd::Real* h_lj_force12, rbmd::Real* h_lj_force6) { '
                       'constexpr rbmd::Id _lj_num_types = 2;\n' + loop + '\n}')
    with tempfile.TemporaryDirectory(prefix='rbmd-lj-host-') as directory:
        work = Path(directory)
        (work / 'production_lj.h').write_text('\n\n'.join(definitions))
        for real in ('float', 'double'):
            executable = work / ('check-' + real)
            subprocess.run([compiler, '-std=c++17', '-O1', '-g',
                            '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                            '-DUSE_DOUBLE=' + str(int(real == 'double')),
                            '-I' + str(ROOT / 'src'), '-I' + str(work),
                            str(ROOT / 'tests/lj_mixing_host.cpp'), '-o', str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
