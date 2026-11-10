import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'04_dynamics_rerun'/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
obs=module('obs','analyze.py');agg=module('agg','aggregate.py');prep=module('postprep','prepare_rerun.py')

class ObservableTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)

    def data(self,style='full'):
        p=self.path/(style+'.data')
        p.write_text('fixture\n\n3 atoms\n\nAtoms # '+style+'\n\n'+''.join(
            f'{i} '+('1 ' if style=='full' else '')+f'{typ} {q} {i} 2 3\n' for i,typ,q in [(1,1,-.8),(2,2,.4),(3,2,.4)]))
        return p

    def trajectory(self,bad_images=False,bad_charge=False):
        rows=''
        for i,typ,q in [(1,1,-.8),(2,2,.4),(3,2,.4)]:
            if bad_charge and i==1:q=-.7
            xu=i+10 if bad_images and i==2 else i
            rows+=f'{i} {typ} {q} {i} 2 3 {xu} 2 3 0 0 0\n'
        return ''.join(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id type q x y z xu yu zu vx vy vz\n'+rows for step in (0,10,20))

    def test_generation_has_only_the_selected_observable(self):
        trajectory=self.path/'trajectory.dump';trajectory.write_text(self.trajectory())
        for mode,name in [('rdf','water'),('dielectric','water'),('structure','lj')]:
            system=ROOT/'systems'/('lj.json' if name=='lj' else name+'.rdf.json')
            s=json.loads(system.read_text());cfg=s['rbmd'];style=cfg['init_configuration']['read_data']['atom_style']
            cfg['init_configuration']['read_data']['file']=str(self.data(style))
            case=self.path/('case-'+mode);case.mkdir();(case/'rbmd.json').write_text(json.dumps(cfg))
            (case/'case.json').write_text(json.dumps(dict(system=name)))
            out=self.path/mode
            cmd=[sys.executable,str(ROOT/'04_dynamics_rerun/prepare_rerun.py'),'--observable',mode,'--system',str(system),'--case',str(case),'--trajectory',str(trajectory),'--output',str(out),'--first','0','--stride','10']
            if mode=='rdf':cmd+=['--radius','8']
            p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            text=(out/'lammps/run.in').read_text()
            for token,expected in [('compute msd_diff',mode=='diffusion'),('compute dip all dipole',mode=='dielectric'),('variable Sx1',mode=='structure'),('compute rdf_all',mode=='rdf')]:
                self.assertEqual(token in text,expected,(mode,token))
            self.assertEqual(json.loads((out/'postprocess.json').read_text())['observable'],mode)

    def test_water_images_and_data_charge_are_validated(self):
        identity=prep.atom_identity(self.data(),'full')
        for bad_images,bad_charge,error in [(True,False,'images'),(False,True,'charge mismatch')]:
            src=self.path/f'src{bad_images}';src.write_text(self.trajectory(bad_images,bad_charge))
            with self.assertRaisesRegex(ValueError,error):
                prep.convert(src,self.path/f'out{bad_images}',0,10,identity=identity,water=True)


    def test_water_dielectric_does_not_require_msd(self):
        meta=dict(observable='dielectric',system='water',units='REAL',n_atoms=3)
        rows=[{'Step':i,'Atoms':3,'Volume':1000,'c_qsum':0,'c_dip[1]':x,'c_dip[2]':0,'c_dip[3]':0} for i,x in enumerate([-1,1,-1,1])]
        result=obs.analyze(rows,meta,300)
        expected=1+(1.602176634e-29)**2/(3*8.8541878128e-12*1000e-30*1.380649e-23*300)
        self.assertAlmostEqual(result['epsilon_r'],expected);self.assertNotIn('D',result)
        grouped=agg.aggregate([result]*5)
        self.assertEqual(grouped['epsilon_r']['n'],5);self.assertEqual(grouped['temperature_K'],300)
        for bad_temperature in (310,None,0,-1,float('nan'),float('inf'),'300',True):
            replicas=[copy.deepcopy(result) for _ in range(5)]
            if bad_temperature is None:replicas[-1].pop('temperature_K')
            else:replicas[-1]['temperature_K']=bad_temperature
            with self.subTest(temperature=bad_temperature),self.assertRaisesRegex(ValueError,'temperature_K'):
                agg.aggregate(replicas)
        rows[0]['c_qsum']=1
        with self.assertRaisesRegex(ValueError,'neutral'):obs.analyze(rows,meta,300)

    def test_structure_retains_actual_k_and_supports_errorbars(self):
        meta=dict(observable='structure',system='lj',units='LJ',n_atoms=10)
        rows=[dict(Step=i,Atoms=10,Lx=10,Ly=20,Lz=30,c_qnorm=10,**{f'v_S{a}{n}':i+1 for a in 'xyz' for n in (1,2,3,4)}) for i in range(4)]
        result=obs.analyze(rows,meta);self.assertNotIn('D',result)
        self.assertAlmostEqual(result['small_k_Szz']['x1']['k'],2*math.pi/10)
        self.assertAlmostEqual(result['small_k_Szz']['y1']['k'],2*math.pi/20)
        self.assertEqual(result['small_k_Szz']['x1']['mean'],2.5)
        replicas=[copy.deepcopy(result) for _ in range(5)]
        self.assertEqual(agg.aggregate(replicas)['small_k_Szz']['x1']['n'],5)
        replicas[-1]['small_k_Szz']['x1']['k']*=2
        with self.assertRaisesRegex(ValueError,'Different k'):agg.aggregate(replicas)

    def test_analysis_cli_uses_metadata_and_detects_missing_frames(self):
        post=self.path/'postprocess.json'
        post.write_text(json.dumps(dict(observable='dielectric',system='water',units='REAL',n_atoms=3,first=0,last=30,stride=10)))
        log=self.path/'log';header='Step Atoms Volume c_qsum c_dip[1] c_dip[2] c_dip[3]\n'
        log.write_text(header+'0 3 1000 0 -1 0 0\n10 3 1000 0 1 0 0\n20 3 1000 0 -1 0 0\n30 3 1000 0 1 0 0\n')
        out=self.path/'result.json'
        cmd=[sys.executable,str(ROOT/'04_dynamics_rerun/analyze.py'),'--postprocess',str(post),'--log',str(log),'--first','0','--last','30','--temperature','300','--output',str(out)]
        p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
        self.assertTrue(out.with_suffix('.csv').exists())
        log.write_text(log.read_text().replace('10 3 1000 0 1 0 0\n',''))
        self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
