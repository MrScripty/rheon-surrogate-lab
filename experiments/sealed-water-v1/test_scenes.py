import copy
import math
import unittest
from collections import Counter
from policy import HERE, decode, read
import scenes


class WaterReservationTests(unittest.TestCase):
    def setUp(self):self.config=decode(read(HERE/'configs/water.json'))

    def test_distinct_whole_scene_splits_and_cubic_geometry(self):
        first=scenes.reserve(self.config);self.assertEqual(first,scenes.reserve(self.config))
        self.assertEqual(Counter(r['split'] for r in first['scenes']),{'train':256,'validation':64,'reserved_test':64})
        self.assertEqual(len({r['seed'] for r in first['scenes']}),384)
        self.assertGreater(len({tuple(r['counts_xyz']) for r in first['scenes']}),300)
        self.assertEqual(len({tuple(r['force']['center_fraction']) for r in first['scenes']}),384)
        for row in first['scenes']:
            h=1/row['cells_per_metre'];extents=[n*h for n in row['counts_xyz']]
            self.assertTrue(all(extents[d]/row['counts_xyz'][d]==h for d in range(3)))
            self.assertAlmostEqual(sum(n*n for n in row['force']['direction']),1)
            self.assertTrue(all(row['counts_xyz'][d]/row['force']['spatial_modes'][d]>=4 for d in range(3)))
        self.assertLess(sum(math.prod(r['counts_xyz'])*64 for r in first['scenes'] if r['split']!='reserved_test'),30*1024**2)

    def test_rejects_different_material_invalid_forcing_and_tampered_reservation(self):
        for mutate in (lambda c:c['material'].update(density_kg_m3=1.25),lambda c:c['force_ranges'].update(frequency_hz=[0,100]),lambda c:c.update(snapshot_steps=[0,63])):
            config=copy.deepcopy(self.config);mutate(config)
            with self.assertRaises(ValueError):scenes.reserve(config)
        reservation=scenes.reserve(self.config);reservation['scenes'][0]['split']='train'
        with self.assertRaises(ValueError):scenes.validate_reservation(reservation)


if __name__=='__main__':unittest.main()
