#!/usr/bin/env python3
"""Compile actual RBE sampling paths with host/MPI shims; no GPU SDK needed."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from run_cvff_bonded_host import extract_definition

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / 'src/force/src/kspace_calculator.cpp').read_text()
    parallel = (ROOT / 'src/parallel/src/rbmd_parallel_until.cpp').read_text()
    # Guard the production wiring as well as executing both samplers below.
    begin = extract_definition(source, 'void KSpaceCalculator::BeginRbeForce(')
    assert 'GetPeriodicBoxForKspace(*_box)' in begin
    assert 'ComputeRbeGaussianSum(force_box, _alpha)' in begin
    assert 'RBEInit(force_box, _alpha, _RBE_P)' in begin
    assert 'PrepareRbeLowModes(force_box)' in begin
    for text in (source, parallel):
        assert '_rbe_replicate_factors' not in text and 'sample_multipliers' not in text
        assert 'BuildRbeSamplingBox' not in text
    for name in ('lj_cut_coul_kspace.cpp', 'cvff.cpp'):
        caller = (ROOT / 'src/force/src' / name).read_text()
        assert 'std::make_unique<KSpaceCalculator>()' in caller
        assert '_kspace_calculator->Execute()' in caller
    header = (ROOT / 'src/parallel/include/rbmd_parallel_until.h').read_text()
    declaration = re.search(r'  void GetRbeRadomM\(.*?;', header, re.S).group()
    sampler = (ROOT / 'src/common/RBEPSample.h').read_text()
    sampler = re.sub(r'^#(?:include|pragma).*$', '', sampler, flags=re.M)
    selector = extract_definition((ROOT / 'src/force/include/force_box_selector.h').read_text(),
                                  'inline Box GetPeriodicBoxForKspace(')
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/rbe_low_modes.h"
namespace rbmd { using Real = REAL_TYPE; using Id = long long; }
struct Real2 { rbmd::Real x,y; };
struct Real3 { rbmd::Real x,y,z; };
struct Box { rbmd::Real _length[3]; };
#define REAL_DATA(v) (&(v).x)
'''
    for macro, fn in dict(SQRT='sqrt', EXP='exp', POW='pow', LOG='log', COS='cos',
                          ROUND='round', ABS='abs', ERF='erf', MIN='fmin').items():
        code += '#define %s std::%s\n' % (macro, fn)
    code += '#define RBMD_PI M_PI\n' + sampler
    code += '''
struct RbmdParallelUntil {
  int _rbe_p_number = 4096;
  struct { Box global_box; } _global_structure_info;
  unsigned int GetRbeGlobalRandomSeed() { return 20261016; }
''' + declaration + '\n};\n'
    code += extract_definition(parallel, 'void RbmdParallelUntil::GetRbeRadomM(')
    code += '''
RbmdParallelUntil parallel;
#define GET_RBMD_PARALLEL (&parallel)
#define MPI_COMM_WORLD 0
#define MPI_RBMD_REAL 0
#define MPI_CHECK(expr) assert((expr) == 0)
int MPI_Comm_rank(int, int* rank) { *rank=0; return 0; }
int MPI_Bcast(void*, int, int, int, int) { return 0; }
struct KSpaceCalculator {
  bool _rbe_low_modes_enabled = false;
  rbmd::Id _rbe_tail_samples = 4096;
  rbmd::RbeLowModes _rbe_low_modes;
  std::mt19937 _rbe_low_generator{20261016};
  std::vector<rbmd::Real> _h_rbe_sample_x, _h_rbe_sample_y, _h_rbe_sample_z, storage;
  rbmd::Real* _h_rbe_samples;
  KSpaceCalculator() : _h_rbe_sample_x(4096), _h_rbe_sample_y(4096),
      _h_rbe_sample_z(4096), storage(4*4096), _h_rbe_samples(storage.data()) {}
  void RBEInit(Box, rbmd::Real, rbmd::Id, bool = true);
};
'''
    code += selector + '\n' + extract_definition(source, 'rbmd::Real ComputeRbeGaussianSum(')
    code += '\n' + extract_definition(source, 'void KSpaceCalculator::RBEInit(')
    code += r'''
int main() {
  for (Box box : {Box{{16,20,24}}, Box{{32,60,24}}, Box{{48,60,72}}}) {
    parallel._global_structure_info.global_box = box;
#ifdef USE_MPI
    Box selected = GetPeriodicBoxForKspace(Box{{8,10,12}});
#else
    Box selected = GetPeriodicBoxForKspace(box);
#endif
    double product = 1;
    for (int d=0; d<3; ++d) {
      assert(selected._length[d] == box._length[d]);
      double sum = 0;
      for(int m=-100;m<=100;++m)
        sum += std::exp(-std::pow(2*M_PI*m/box._length[d],2)/(4*.08));
      product *= sum;
    }
    assert(std::abs(ComputeRbeGaussianSum(selected, .08) - (product-1)) < 2e-6*product);
    for (bool low : {false, true}) {
      KSpaceCalculator calc;
      calc._rbe_low_modes_enabled = low;
      calc._rbe_low_modes.Setup({double(box._length[0]),double(box._length[1]),double(box._length[2])}, .08, 0);
      calc.RBEInit(selected, .08, 4096);
      int outside[3] = {}, unit[3] = {};
      for(int i=0;i<4096;++i) {
        bool nonzero = false;
        for(int d=0;d<3;++d) {
          double value = calc.storage[d*4096+i];
          assert(std::isfinite(value) && value == std::round(value));
          int n = int(value);
          nonzero |= n != 0;
          outside[d] += n % 3 != 0;
          unit[d] += std::abs(n) == 1;
        }
        assert(nonzero);
      }
      for(int d=0;d<3;++d) assert(outside[d] > 100 && unit[d] > 20);
    }
  }
  std::cout << "PASS full-box S, global box, MH and split samplers, non-replica modes\n";
}
'''
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    assert compiler, 'C++17 compiler required'
    with tempfile.TemporaryDirectory(prefix='rbmd-g06-') as directory:
        path = Path(directory)
        (path / 'test.cpp').write_text(code)
        for real in ('float', 'double'):
            for mpi in (False, True):
                exe = path / ('test-' + real + str(mpi))
                command = [compiler, '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined',
                           '-fno-omit-frame-pointer', '-DREAL_TYPE=' + real, '-I' + str(ROOT / 'src')]
                if mpi:
                    command.append('-DUSE_MPI')
                subprocess.run(command + [str(path / 'test.cpp'), '-o', str(exe)], check=True)
                subprocess.run([str(exe)], check=True, timeout=60)
                print('PASS', real, 'MPI-shim' if mpi else 'non-MPI', flush=True)


if __name__ == '__main__':
    main()
