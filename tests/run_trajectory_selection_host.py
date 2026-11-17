#!/usr/bin/env python3
"""Exercise production selection/config code in all precision/ID modes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
cxx = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
with tempfile.TemporaryDirectory(prefix='trajectory-host-') as temp:
    for precision in (0, 1):
        for ids in (0, 1):
            exe = str(Path(temp) / f'test-{precision}-{ids}')
            subprocess.run([cxx, '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined',
                            f'-DUSE_DOUBLE={precision}', f'-DUSE_64BIT_IDS={ids}',
                            '-I'+str(ROOT/'src'), str(ROOT/'tests/trajectory_selection_host.cpp'),
                            '-o', exe], check=True)
            subprocess.run([exe], check=True)
