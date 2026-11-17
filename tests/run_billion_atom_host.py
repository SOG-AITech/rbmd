#!/usr/bin/env python3
"""Host regressions using production code; no claim of GPU/MPI integration."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
def main():
    cxx = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not cxx:
        raise SystemExit('C++17 compiler required')
    with tempfile.TemporaryDirectory(prefix='rbmd-billion-') as tmp:
        for precision in (0, 1):
            exe = str(Path(tmp) / ('host-' + str(precision)))
            subprocess.run([cxx, '-std=c++17', '-O1', '-g',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                '-DUSE_64BIT_IDS=1', '-DUSE_DOUBLE=' + str(precision),
                '-I'+str(ROOT/'src'), '-I'+str(ROOT/'src/reader/include'),
                str(ROOT/'tests/billion_atom_host.cpp'), '-o', exe], check=True)
            subprocess.run([exe], check=True)
            mpi_exe = exe + '-mpi-protocol'
            subprocess.run([cxx, '-std=c++17', '-O1', '-g', '-pthread',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                '-DUSE_64BIT_IDS=1', '-DUSE_DOUBLE=' + str(precision),
                '-DUSE_MPI', '-DREADER_ENABLE_MPI',
                '-I'+str(ROOT/'tests/mpi_host_stub'),
                '-I'+str(ROOT/'src'), '-I'+str(ROOT/'src/reader/include'),
                str(ROOT/'tests/reader_mpi_host.cpp'), '-o', mpi_exe], check=True)
            subprocess.run([mpi_exe], check=True, timeout=30)

if __name__ == '__main__': main()
