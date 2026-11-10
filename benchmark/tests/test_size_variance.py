import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('size_collect',ROOT/'03_size_variance/collect.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)

class SizeVariance(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for i,n in enumerate((2,16)):
            d=self.root/f'L{i}';d.mkdir()
            (d/'manifest.json').write_text(json.dumps(dict(system=f'L{i}',N=n,status='ADOPTED_TEST_SOURCE')))
            L=10*(i+1)
            (d/'terminal.data').write_text(f'fixture\n{n} atoms\n\n0 {L} xlo xhi\n0 {L} ylo yhi\n0 {L} zlo zhi\n\nPair Coeffs\n\n1 1 1\n2 1 1\n')
        self.out=self.root/'out'
    def prepare(self,ok=True):
        args=[sys.executable,str(ROOT/'03_size_variance/prepare.py'),'--system',str(ROOT/'systems/lj.json'),'--data-root',str(self.root),'--data','L0/terminal.data','L1/terminal.data','--P','200','--p','30','--alpha','.024','--rs','4','--rc','2','--samples','3','--output',str(self.out)]
        r=subprocess.run(args,capture_output=True,text=True)
        self.assertEqual(r.returncode==0,ok,r.stderr)
    def test_only_frozen_force_cases_and_double_references(self):
        self.prepare();cases=list(self.out.rglob('case.json'));self.assertEqual(len(cases),10)
        self.assertEqual(json.loads((self.out/'manifest.json').read_text())['methods'],['rbe','combined'])
        self.assertFalse(list(self.out.rglob('rbl')))
        for f in cases:
            meta=json.loads(f.read_text());cfg=json.loads(f.with_name('rbmd.json').read_text());diag=cfg['execution']['diagnostics']
            self.assertEqual(cfg['execution']['num_steps'],0)
            self.assertFalse(diag['thermostat_heat']);self.assertFalse(diag['energy'])
            if meta['engine']=='rbmd':
                self.assertEqual(diag['fixed_samples'],3);self.assertFalse(diag['forces']);self.assertEqual(diag['reference_id_offset'],-1)
                self.assertTrue(diag['reference_force_file'].endswith('/pppm_1e-07/reference.dump'))
                self.assertEqual(cfg['hyper_parameters']['coulomb']['type'],'RBE')
                self.assertNotIn('kmax',cfg['hyper_parameters']['coulomb'])
                self.assertEqual(cfg['hyper_parameters']['coulomb']['alpha'],.024)
            else:
                self.assertEqual(meta['lammps_role'],'reference')
                text=f.with_name('run.in').read_text();self.assertIn('kspace_modify order 7',text);self.assertIn('run 0',text);self.assertNotIn(' nve',text);self.assertNotIn('thermostat',text)
    def test_unfinished_source_rejected_without_output(self):
        p=self.root/'L1/manifest.json';s=json.loads(p.read_text());s['status']='PREPARING_LATEST_VALIDATED_NOT_FINAL';p.write_text(json.dumps(s))
        self.prepare(False);self.assertFalse(self.out.exists())
    def test_collect_separates_error_variance_and_rejects_alpha_change(self):
        self.prepare();manifest=json.loads((self.out/'manifest.json').read_text())
        for entry in manifest['inputs']:
            for method in collector.METHODS:
                case=self.out/f"N{entry['N']}"/'rs4'/method;cfg=json.loads((case/'rbmd.json').read_text())
                values=dict(parameters=cfg['hyper_parameters'],solver_settings=dict(alpha=.024,alpha_source='manual'),n_atoms=entry['N'],n_samples=3,reference_converged=True,mse=5.,rms_force_error=5**.5,rms_bias=1.,variance_population=4.,variance_unbiased=6.)
                (case/'force_statistics.json').write_text(json.dumps(values))
                (case/'run_receipt.json').write_text('{"exit_code":0}')
                (case/'preflight.json').write_text(json.dumps(dict(status='PASS',data_sha256=entry['sha256'],n_atoms=entry['N'])))
        result=collector.collect(self.out);self.assertEqual(set(result['methods']),set(collector.METHODS))
        self.assertEqual(result['methods']['rbe']['rows'][0]['mse'],5.)
        f=self.out/'N2/rs4/rbe/force_statistics.json';s=json.loads(f.read_text());s['solver_settings']['alpha']=.025;f.write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,'alpha'):collector.collect(self.out)
if __name__=='__main__':unittest.main()
