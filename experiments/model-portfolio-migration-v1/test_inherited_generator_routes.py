"""Route fixtures from original geometry assets and constructor-readiness facts."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from inherited_generator_cuda import ROOT, read, route


class CapturedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        readiness = read(ROOT/'docs/plans/model-portfolio-migration-v1/reports/temporal-GPU-construction-readiness-01.json')
        fine = next(r for r in readiness['rows'] if r['identifier'] == 'b2-endpoint-decoder')
        full = next(r for r in readiness['rows'] if r['identifier'] == 'b1-fixed-viscosity')
        path = ROOT/'runs/full-ai-next-priorities-v1/b2-endpoint-decoder-project-01/expert-assets/pressure/configuration.json'
        geometry = {tuple(r['counts_zyx']):(f'g{i}',r['tail_gain_key'] is not None)
                    for i,r in enumerate(read(path)['geometries'])}
        cls.fine = SimpleNamespace(geometry=geometry)
        for shape,(prefix,_) in geometry.items():
            if math.prod(shape) in fine['fine_unit_query_counts']:
                setattr(cls.fine,prefix+'_fine_matrix',object())
        cls.full = SimpleNamespace(routes={s:('captured-map' if math.prod(s) in full['full_unit_query_counts'] else None)
            for s in geometry})

    def test_both_rectangles_use_the_generated_fine_maps(self):
        self.assertEqual(route(self.fine,[9,11,11]),'fine_F64_unit_map')
        self.assertEqual(route(self.fine,[15,6,8]),'fine_F64_unit_map')

    def test_cubes_exercise_distinct_complete_map_and_fallback_routes(self):
        self.assertEqual(route(self.fine,[16,16,16]),'fine_F32_CNN')
        self.assertEqual(route(self.full,[16,16,16]),'complete_F64_unit_map')
        self.assertEqual(route(self.full,[32,32,32]),'core_CNN_fallback')

    def test_unsupported_geometry_reports_the_correct_owner(self):
        with self.assertRaisesRegex(ValueError,'^Fine pressure core has no captured geometry$'):
            route(self.fine,[17,19,23])
        with self.assertRaisesRegex(ValueError,'^Complete pressure map has no captured geometry$'):
            route(self.full,[17,19,23])


if __name__ == '__main__':
    unittest.main()
