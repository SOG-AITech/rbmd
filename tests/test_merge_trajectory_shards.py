import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('merge', Path(__file__).resolve().parents[1]/'scripts/merge_trajectory_shards.py')
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def frame(step, rows):
    return (f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n'
            'ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n'
            'ITEM: ATOMS id type xu yu zu\n' + ''.join(rows))


class MergeTests(unittest.TestCase):
    def test_empty_rank_and_migrating_64bit_id(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'a').write_text(frame(0, ['10000000000 3 1 2 3\n']) + frame(10, []))
            (p/'b').write_text(frame(0, ['1 1 0 0 0\n']) + frame(10, ['10000000000 3 2 2 3\n', '1 1 0 0 0\n']))
            self.assertEqual(merge.merge([p/'a', p/'b'], p/'out'), 2)
            with (p/'out').open() as f:
                for _ in range(2):
                    self.assertEqual([r.split()[0] for r in merge.read_frame(f)[4]], ['1', '10000000000'])
                self.assertIsNone(merge.read_frame(f))
            with self.assertRaises(FileExistsError):
                merge.merge([p/'a', p/'b'], p/'out')

    def test_reject_inconsistent_or_duplicate_shards(self):
        row = '1 1 0 0 0\n'
        for b in (frame(1, []), frame(0, [row]), '', frame(0, ['2 1 0\n'])):
            with self.subTest(b=b), tempfile.TemporaryDirectory() as temp:
                p = Path(temp)
                (p/'a').write_text(frame(0, [row]))
                (p/'b').write_text(b)
                with self.assertRaises(ValueError):
                    merge.merge([p/'a', p/'b'], p/'out')
                with self.assertRaises(ValueError):
                    merge.merge([p/'a'], p/'a')


if __name__ == '__main__':
    unittest.main()
