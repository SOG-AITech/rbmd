#!/usr/bin/env python3
import argparse,hashlib,json,os,re,shlex,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,m):
 if not x:raise RuntimeError(m)
def extract(t,spec):
 start,end=spec['start'],spec['end'];need(t.count(start)==1,'nonunique fragment anchor '+spec['name']);a=t.index(start);b=t.rfind(end)if spec.get('last_end')else t.index(end,a+len(start));v=t[a:b].strip()+'\n';need(hashlib.sha256(v.encode()).hexdigest()==spec['sha256'],'fragment SHA mismatch '+spec['name']);return v
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,type=Path);ap.add_argument('--build',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();need(os.environ.get('SLURM_JOB_ID'),'Require an allocated Slurm GPU job');s,b,o=a.source.resolve(),a.build.resolve(),a.output.resolve();o.mkdir(parents=True,exist_ok=False)
 r={'schema':'fixedp-v5-real-gpu-core-alias-input/v1','status':'NOT_RUN','job':os.environ['SLURM_JOB_ID'],'source':str(s),'build':str(b),'scope':'Actual GPU legacy core producer + v5 Prepare contract; not MPI/MD/RDF/performance or promotion'}
 try:
  pins=json.loads((H/'pins.json').read_text());r['source_sha256']={}
  for rel,expected in pins['production_files'].items():actual=sha(s/rel);need(actual==expected,'source SHA mismatch '+rel);r['source_sha256'][rel]=actual
  t=(s/pins['neighbor_tu']).read_text();chunks=[extract(t,x)for x in pins['fragments']]
  prefix='#include "full_neighbor_list_op.h"\n#include "common/atomic_count.h"\n#include "rbl_full_neighbor_list_op.h"\n#include "common/rbl_fixedp_uniform.hpp"\n#include <limits>\n'
  cpp=o/'actual_v5_core_alias.hip.cpp';cpp.write_text(prefix+'namespace op {\n'+''.join(chunks)+'\n}\n'+(H/'fixture_main.cpp').read_text())
  f=b/'src/neighbor_list/CMakeFiles/neighbor_list.dir/flags.make';t=f.read_text().replace('\\\n',' ');flags=[]
  for name in ('DEFINES','INCLUDES','FLAGS'):
   m=re.search(r'^CXX_'+name+r'\s*=\s*(.*)$',t,re.M);need(m,'Missing CXX_'+name);flags+=shlex.split(m.group(1))
  cache=b/'CMakeCache.txt';m=re.search(r'^CMAKE_CXX_COMPILER:(?:FILEPATH|STRING)=(.+)$',cache.read_text(),re.M);need(m,'Missing compiler');exe=o/'actual_v5_core_alias';cmd=[m.group(1).strip()]+flags+['-x','hip',str(cpp),'-o',str(exe)];r.update(command=cmd,flags_sha256=sha(f),cmake_cache_sha256=sha(cache),generated_sha256=sha(cpp),fixture_sha256=sha(H/'fixture_main.cpp'))
  c=subprocess.run(cmd,cwd=b,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(o/'compile.log').write_text(c.stdout);need(c.returncode==0,'HIP compile failed: compile.log');r['executable_sha256']=sha(exe)
  c=subprocess.run([str(exe)],cwd=o,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(o/'gpu.log').write_text(c.stdout);r['gpu_exit']=c.returncode;need(c.returncode==0 and 'PASS_REAL_HIP_FIXEDP_V5_CORE_ALIAS_INPUTS'in c.stdout,'GPU fixture failed: gpu.log')
  for rel,expected in pins['production_files'].items():need(sha(s/rel)==expected,'source changed while testing '+rel)
  need(sha(f)==r['flags_sha256']and sha(cache)==r['cmake_cache_sha256'],'build flags changed while testing');r.update(status='PASS_REAL_HIP_V5_CORE_ALIAS_INPUTS',gpu_log_sha256=sha(o/'gpu.log'))
 except Exception as e:r.update(status='FAILED',error=str(e))
 (o/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return 0 if r['status']=='PASS_REAL_HIP_V5_CORE_ALIAS_INPUTS'else 1
if __name__=='__main__':sys.exit(main())
