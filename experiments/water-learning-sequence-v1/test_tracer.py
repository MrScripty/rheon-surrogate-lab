"""Ensure motion diagnostics distinguish transport from repainting a fixed blob."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from policy import canonical
from tracer_check import measure


class TracerTests(unittest.TestCase):
    def session(self, shifted):
        z, y, x = np.indices((9, 9, 9))
        fields = [np.exp(-((x-3)**2+(y-4)**2+(z-4)**2)/1.5),
                  np.exp(-((x-(4 if shifted else 3))**2+(y-4)**2+(z-4)**2)/1.5)]
        return {
            'schema': 'rheon-water-training-timeline-v1', 'trajectory_id': 'synthetic-display-check',
            'counts_xyz': [9, 9, 9], 'cell_size_m': 0.125,
            'frames': [{'index': i, 'time_s': (i+1)/60, 'tracer': f.reshape(-1).tolist(),
                        'velocity': [[0.125, 0, 0]]*729} for i, f in enumerate(fields)],
            'provenance': {'native_steps_qualified': 2, 'retained_frame_parity': {}},
        }

    def check(self, session):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'session.json'
            path.write_bytes(canonical(session))
            return measure(path)

    def test_changing_values_without_transport_does_not_pass_visibility(self):
        session = self.session(False)
        session['frames'][1]['tracer'] = [value*0.8 for value in session['frames'][1]['tracer']]
        self.assertFalse(self.check(session)['visible_motion'])

    def test_one_cell_transport_is_measured_in_voxels(self):
        result = self.check(self.session(True))
        self.assertTrue(result['visible_motion'])
        self.assertAlmostEqual(result['centroid_displacement_cells_xyz'][0], 1.0, places=3)
        self.assertAlmostEqual(result['centroid_displacement_cells_xyz'][1], 0.0, places=12)

    def test_scene_reset_is_rejected(self):
        session = copy.deepcopy(self.session(True))
        session['frames'][1]['index'] = 0
        with self.assertRaisesRegex(ValueError, 'tracer_order'):
            self.check(session)

    def test_geometry_budget_is_checked_before_allocating_positions(self):
        session = self.session(False)
        session['counts_xyz'] = [1000000]*3
        with self.assertRaisesRegex(ValueError, 'tracer_budget'):
            self.check(session)


if __name__ == '__main__':
    unittest.main()
