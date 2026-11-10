import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('reduce_native', Path(__file__).with_name('reduce_native.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReductionTest(unittest.TestCase):
    def test_bad_images_never_produce_a_diffusion_coefficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / 'small/lj/rep1/rbmd'
            case.mkdir(parents=True)
            folder = root / 'post'
            folder.mkdir()
            checks = root / 'checks-round1'
            checks.mkdir()
            meta = dict(system='lj', source_engine='rbmd', n_atoms=2, source_case=str(case),
                        last=100000, stride=100, timestep=.001, dr=.5, radius=1,
                        rdf_exclusions='none')
            (folder / 'postprocess.json').write_text(json.dumps(meta))
            check = dict(frames=1001, finite_values=True, complete_unique_ids=True,
                         max_displacement_between_frames=271)
            checkfile = checks / 'lj-rbmd.json'
            checkfile.write_text(json.dumps(check))
            modes = [f'v_S{a}{m}' for m in range(1, 5) for a in 'xyz']
            log = ['Step Atoms Volume Lx Ly Lz c_msd1[4] c_msd2[4] c_qnorm ' + ' '.join(modes)]
            for step in range(0, 100001, 100):
                msd = 12*step*.001
                log.append(f'{step} 2 19902511 271 271 271 {msd} {msd} 2 ' + ' '.join(['.1']*12))
            (folder / 'log.lammps').write_text('\n'.join(log))
            (folder / 'rdf.dat').write_text('\n'.join(
                f'{s} 2\n1 .25 1 0 1 0 1 0\n2 .75 1 0 1 0 1 0' for s in range(0, 100001, 100)))
            result = module.reduce(folder)
            self.assertIsNone(result['diffusion']['1']['D'])
            self.assertEqual(result['diffusion']['1']['status'], 'INVALID_UNWRAPPED_IMAGES')
            check['max_displacement_between_frames'] = 1
            checkfile.write_text(json.dumps(check))
            result = module.reduce(folder)
            self.assertAlmostEqual(result['diffusion']['1']['D'], 2)
            self.assertEqual(result['rdf']['frames'], 801)

    def test_constant_dipole_has_unit_dielectric(self):
        rows = [{'Volume': 1000, 'c_dip_local[1]': 3, 'c_dip_local[2]': -2,
                 'c_dip_local[3]': 7} for _ in range(3)]
        self.assertEqual(module.epsilon(rows, 298), 1)


if __name__ == '__main__':
    unittest.main()
