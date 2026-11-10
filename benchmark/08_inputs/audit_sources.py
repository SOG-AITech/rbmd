#!/usr/bin/env python3
"""Inventory candidate inputs, historical replication, and exact source hashes; never certify equilibration."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path

SOURCE_FILES=('src/force/include/force_box_selector.h','src/force/src/kspace_calculator.cpp',
 'src/force/src/kspace_calculator_op/rocm/kspace_calculator_op.hip.cu',
 'src/parallel/src/rbmd_parallel_until.cpp','src/parallel/include/rbmd_parallel_until.h')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def header(path):
    with path.open(errors='replace') as f:text=f.read(65536)
    match=re.search(r'^\s*(\d+)\s+atoms\s*$',text,re.M)
    return int(match[1]) if match else None

def inventory(root):
    data=[];runs=[];errors=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(x in path.parts for x in ('.git','__pycache__')):continue
        try:
            if path.suffix=='.data' or path.name.startswith('data.'):
                n=header(path)
                data.append(dict(path=str(path),atoms=n,bytes=path.stat().st_size,
                    target_candidate=n in (2703,72981,976200),equilibration='UNVERIFIED'))
            elif path.suffix=='.json' and path.stat().st_size<10_000_000:
                text=path.read_text()
                text=re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',lambda m:m[0] if m[0].startswith(chr(34)) else '',text,flags=re.S)
                obj=json.loads(text)
                if not isinstance(obj,dict):continue
                cfg=obj.get('rbmd',obj)
                read=cfg.get('init_configuration',{}).get('read_data')
                if isinstance(read,dict):runs.append(dict(path=str(path),format='rbmd',replicate=read.get('replicate',[1,1,1]),data=read.get('file'),evidence='configuration only; execution not established'))
            elif path.suffix in ('.in','.slurm') or path.name.startswith(('in.','log.')):
                commands=[]
                with path.open(errors='replace') as f:
                    for number,line in enumerate(f,1):
                        if re.match(r'^\s*(replicate|read_data|read_restart)\s',line):commands.append(dict(line=number,text=line.strip()))
                if commands:runs.append(dict(path=str(path),format='lammps',commands=commands,evidence='input or log; inspect referenced include/variable files'))
        except (OSError,ValueError,AttributeError) as e:errors.append(dict(path=str(path),error=str(e)))
    return dict(inputs=data,historical_runs=runs,read_errors=errors)

def source_record(root):
    hashes={name:digest(root/name) if (root/name).is_file() else None for name in SOURCE_FILES}
    cpp=root/SOURCE_FILES[1];text=cpp.read_text() if cpp.is_file() else ''
    def git(*args):
        p=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else None
    return dict(root=str(root),sha256=hashes,commit=git('rev-parse','HEAD'),
        working_tree=git('status','--short'),full_box_markers_present=all(x in text for x in ('GetPeriodicBoxForKspace(*_box)','RBEInit(force_box','ComputeRbeGaussianSum(force_box')),
        binary_contains_these_sources='UNVERIFIED: requires matching build provenance and runtime test')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root',type=Path,required=True);p.add_argument('--benchmark-root',type=Path)
    p.add_argument('--expected-source',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bench=a.benchmark_root or a.code_root/'benchmark'
    if not a.code_root.is_dir() or not bench.is_dir():p.error('Code/benchmark directory missing')
    result=inventory(bench)
    for key in ('inputs','historical_runs','read_errors'):
        result[key]=[row for row in result[key] if any(token in json.dumps(row).lower() for token in ('peo','litfsi')) == False]
    result['scope']='lj-water'
    result['source']=source_record(a.code_root)
    if a.expected_source:
        expected=json.loads(a.expected_source.read_text())['source']['sha256']
        result['source_matches_expected']=all(v is not None and result['source']['sha256'].get(k)==v for k,v in expected.items())
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(a.output)
if __name__=='__main__':main()
