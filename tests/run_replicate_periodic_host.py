#!/usr/bin/env python3
"""Production replication helpers; float/double and 32/64-bit IDs."""
import os,shutil,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cxx=os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
with tempfile.TemporaryDirectory() as tmp:
 for real in (0,1):
  for ids in (0,1):
   exe=str(Path(tmp)/f'test-{real}-{ids}')
   subprocess.run([cxx,'-std=c++17','-fsanitize=address,undefined','-g',f'-DUSE_DOUBLE={real}',f'-DUSE_64BIT_IDS={ids}','-I'+str(root/'src'),'-I'+str(root/'src/reader/include'),str(root/'tests/replicate_periodic_host.cpp'),'-o',exe],check=True)
   subprocess.run([exe],check=True)
# Both reader paths must use the default and chain fix, not just the helper.
s=(root/'src/reader/src/format/lammps/lammps_full_reader.cpp').read_text()
assert 'replicate_bond_periodic_ = true;' in s
assert s.count('? BondPeriodicChainEnd(seed, atom_lookup, dihedral.atom3,')==2
assert 'return !replicate_bond_periodic_ &&' in s
