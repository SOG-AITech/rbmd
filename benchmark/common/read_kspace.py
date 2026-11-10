"""Read actual solver settings from stdout, leaving missing fields explicit."""
import math
import re

NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

def settings(text):
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    result={}
    for key,pattern in {
        'alpha':rf'\balpha\s*:\s*({NUMBER})',
        'gewald':rf'G vector \(1/distance\)\s*=\s*({NUMBER})',
    }.items():
        matches=re.findall(pattern,text)
        if matches:
            values=list(map(float,matches))
            if any(not math.isfinite(x) or x<=0 for x in values):raise ValueError(f'Invalid {key}')
            if any(not math.isclose(x,values[0],rel_tol=1e-6) for x in values):raise ValueError(f'{key} changed within log')
            result[key]=values[-1]
    source=re.findall(r'alpha_source\s*:\s*(auto|manual)',text)
    if source:
        if len(set(source))!=1:raise ValueError('Mixed alpha sources')
        result['alpha_source']=source[-1]
    grid=re.findall(r'\bgrid\s*=\s*(\d+)\s+(\d+)\s+(\d+)',text)
    order=re.findall(r'stencil order\s*=\s*(\d+)',text)
    if grid:result['pppm_grid']=list(map(int,grid[-1]))
    if order:result['pppm_order']=int(order[-1])
    return result
