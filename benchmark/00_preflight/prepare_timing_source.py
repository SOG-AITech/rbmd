#!/usr/bin/env python3
"""Copy RBMD source and remove diagnostic-only forced waits in production mode.
Mandatory communication/stream-dependency synchronization remains intact.
This is an isolated benchmark build, not an in-place engine update.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

def guarded(text, expected):
    old='rbmd::revision::DeviceSynchronize();'
    if text.count(old)!=expected:raise ValueError('Source changed: inspect timing sync sites before patching')
    return text.replace(old,'if (rbmd::revision::GetOptions().isolated_timing) '+old)

def sample_diagnostics(text):
    anchor='return _neighbor_list;'
    if text.count(anchor)!=2 or text.count('extern rbmd::Id test_current_step;')!=1:raise ValueError('RBL builder changed; inspect diagnostic placement')
    helper=Path(__file__).with_name('rbl_sample_diagnostics.inc').read_text()
    return text.replace('extern rbmd::Id test_current_step;',helper+'\nextern rbmd::Id test_current_step;').replace(
        anchor,'RecordBenchmarkRblSamples(*_neighbor_list, ActiveAtomCount(), test_current_step);\n  '+anchor)

def patch(source,dest):
    if dest.exists():raise ValueError('Destination exists; keep source/build evidence immutable')
    changes={}
    required = {
        'src/simulate_pipeline/src/simulate.cpp': 'if (diagnostics.isolated_timing)',
        'src/neighbor_list/src/neighbor_list_builder/rbl_full_neighbor_list_builder.cpp': 'RecordBenchmarkRblSamples',
        'src/simulate_pipeline/src/ensemble.cpp': 'GetOptions().isolated_timing',
    }
    for name, marker in required.items():
        text = (source/name).read_text()
        if marker not in text:
            raise ValueError('Use the public paper branch with integrated diagnostics: ' + name)
        changes[name] = (text, text)
    dest.mkdir(parents=True)
    shutil.copy2(source/'CMakeLists.txt',dest/'CMakeLists.txt')
    (dest/'scripts').mkdir()
    shutil.copy2(source/'scripts/postprocess_freud.py',dest/'scripts/postprocess_freud.py')
    for directory in ('cmake','src','tools'):
        shutil.copytree(source/directory,dest/directory,ignore=shutil.ignore_patterns('*.orig','*.rej','._*','__pycache__'))
    diff=[];hashes={}
    for name,(old,new) in changes.items():
        (dest/name).write_text(new)
        hashes[name]={'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest()}
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (dest/'timing.patch').write_text(''.join(diff));(dest/'timing_source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(dest)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();patch(a.source,a.destination)
if __name__=='__main__':main()
