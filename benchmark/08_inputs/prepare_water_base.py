#!/usr/bin/env python3
"""Attach canonical SPC/E coefficients to the archived 2703-atom seed without changing coordinates."""
import argparse,json,re
from pathlib import Path
from audit_sources import digest,header

def section(text,name):
    match=re.search(r'^'+re.escape(name)+r'(?:\s*#[^\n]*)?\s*\n(.*?)(?=^[A-Za-z]|\Z)',text,re.M|re.S)
    if not match:raise ValueError('Missing section '+name)
    return match.group(0)

def materialize(base,coefficients,output):
    if header(base)!=2703 or header(coefficients)!=72981:raise ValueError('Expected canonical 2703/72981 water inputs')
    text=base.read_text();source=coefficients.read_text()
    def values(s):return [line.split('#',1)[0].split() for line in s.splitlines()[1:] if line.split('#',1)[0].strip()]
    if values(section(text,'Masses'))!=values(section(source,'Masses')):raise ValueError('Water masses differ')
    names=('Pair Coeffs','Bond Coeffs','Angle Coeffs')
    if any(re.search(r'^'+n,text,re.M) for n in names):raise ValueError('Seed already contains coefficients; inspect instead of overwriting')
    blocks='\n'.join(section(source,n) for n in names)
    if len(re.findall(r'^Atoms\b',text,re.M))!=1:raise ValueError('Expected one Atoms section')
    result=re.sub(r'^Atoms\b',lambda m:blocks+'\nAtoms',text,count=1,flags=re.M)
    with output.open('x') as f:f.write(result)
    return dict(base=str(base),base_sha256=digest(base),coefficients=str(coefficients),coefficients_sha256=digest(coefficients),output_sha256=digest(output),coordinates_unchanged=True,equilibration='archive labels equilibrated; original log and full-box replica decorrelation not established')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    record=materialize(a.data_root/'h2o_rdf.data',a.data_root/'h2o.data',a.output/'water2703.data')
    (a.output/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(a.output/'water2703.data')
if __name__=='__main__':main()
