#!/usr/bin/env python3
"""Check production per-atom kernel arithmetic on the host, without a GPU SDK.

Extracts the actual kernels (not copies of their formulas), compiles them with
one-thread block/reduction shims, and checks finite-difference forces/virials and
owned/ghost partitions. This does not test GPU scheduling or MPI communication.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/force/src/cvff_op/rocm/cvff_op.hip.cu'


def extract_definition(source, marker):
    start = source.index(marker)
    body = source.index('{', start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        raise SystemExit('A C++17 host compiler is required')
    source = SOURCE.read_text()
    names = ('ComputeBondForcePerAtom', 'ComputeAngleForcePerAtom')
    kernels = '\n\n'.join(extract_definition(
        source, '__global__ void ' + name + '(') for name in names)
    # Also compile the production op signatures/forwarding calls. Only launch
    # syntax is removed; these wrappers are not used to simulate GPU scheduling.
    declarations = (SOURCE.parents[1] / 'cvff_op.h').read_text()
    ops = []
    for name in names:
        op = name + 'Op'
        declaration = extract_definition(
            declarations, 'struct ' + op + '<device::DEVICE_GPU>')
        definition = extract_definition(
            source, 'void ' + op + '<device::DEVICE_GPU>::operator()(')
        definition = re.sub(r'<<<.*?>>>', '', definition, flags=re.DOTALL)
        ops.append('template<typename> struct ' + op + ';\ntemplate<>\n' +
                   declaration + ';\n' + definition)
    kernels += '\nnamespace op {\n' + '\n'.join(ops) + '\n}\n'
    with tempfile.TemporaryDirectory(prefix='rbmd-cvff-host-') as directory:
        work = Path(directory)
        (work / 'production_kernels.h').write_text(kernels)
        for real in ('float', 'double'):
            executable = work / ('check-' + real)
            subprocess.run([
                compiler, '-std=c++17', '-O1', '-g',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                '-DUSE_64BIT_IDS=1', '-DUSE_DOUBLE=' + str(int(real == 'double')),
                '-I' + str(ROOT / 'src'), '-I' + str(work),
                str(ROOT / 'tests/cvff_bonded_host.cpp'), '-o', str(executable),
            ], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
