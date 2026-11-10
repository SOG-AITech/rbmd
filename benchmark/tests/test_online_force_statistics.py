import importlib.util,json,math,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('online',ROOT/'03_size_variance/analyze_online.py')
online=importlib.util.module_from_spec(spec);spec.loader.exec_module(online)
class OnlineStatistics(unittest.TestCase):
    def test_decomposition_and_rejections(self):
        s=dict(schema=1,n_samples=3,n_atoms=2,mse=58.,rms_force_error=math.sqrt(58),relative_rms_force_error=1.,rms_bias=math.sqrt(50),variance_population=8.,variance_unbiased=12.)
        online.validate_moments(s,3)
        for field,value in [('variance_population',9.),('variance_unbiased',8.),('mse',float('nan')),('n_samples',2)]:
            with self.assertRaises(ValueError):online.validate_moments(dict(s,**{field:value}),3)
    def test_reference_stream_no_frame_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'reference.dump'
            p.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id x y z fx fy fz\n1 0 0 0 1 0 0\n2 1 0 0 0 2 0\n')
            stream=online.dump_rows(p);self.assertEqual(next(stream)[0],2)
            self.assertEqual(next(stream)[0],1);self.assertEqual(next(stream)[0],2)
            with self.assertRaises(StopIteration):next(stream)
            for old,new in [('2 1 0 0','1 1 0 0'),('0 2 0','0 nan 0')]:
                original=p.read_text();p.write_text(original.replace(old,new))
                with self.assertRaises(ValueError):list(online.dump_rows(p))
                p.write_text(original)
if __name__=='__main__':unittest.main()
