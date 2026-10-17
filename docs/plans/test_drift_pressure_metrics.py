"""Focused checks against pressure vectors, not against copied chart formulas."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from drift_pressure_metrics import enrich_drift_pressure, pressure_magnitudes


class PressureMagnitudeTests(unittest.TestCase):
    def test_independent_pressure_error_is_bounded_and_not_same_request_error(self):
        reference = [0, 3, 4]
        candidate = [0, 3.03, 4.04]
        same_request_reference = [0, 3.02, 4.01]
        norm = lambda a: math.sqrt(sum(x*x for x in a))
        distance = lambda a, b: norm([x-y for x, y in zip(a, b)])
        error = distance(candidate, reference) / norm(reference)
        same_distance = distance(candidate, same_request_reference)
        frame = dict(pressure_comparison_scope='independent_trajectories', relative_pressure_difference=error,
                     diagnostic_attempt=dict(same_request_pressure_comparison=dict(status='available',
                         l2=same_distance, relative_l2=same_distance/norm(same_request_reference))))
        row, reason = pressure_magnitudes(frame, [3, 1, 1])
        self.assertEqual(reason, 'bounded estimate')
        actual_reference = norm(reference)/math.sqrt(3)
        actual_error = distance(candidate, reference)/math.sqrt(3)
        self.assertLessEqual(row[2], actual_reference)
        self.assertGreaterEqual(row[3], actual_reference)
        self.assertLessEqual(error*row[2], actual_error)
        self.assertGreaterEqual(error*row[3], actual_error)
        self.assertNotAlmostEqual(row[0], same_distance/math.sqrt(3))

    def test_unsupported_norms_stay_missing(self):
        frame = dict(pressure_comparison_scope='independent_trajectories', relative_pressure_difference=0.01)
        self.assertIsNone(pressure_magnitudes(frame, [16, 16, 16])[0])
        frame['diagnostic_attempt'] = dict(same_request_pressure_comparison=dict(status='available',l2=1,relative_l2=0.1))
        for error in (1, 2, -1, float('nan')):
            frame['relative_pressure_difference'] = error
            self.assertIsNone(pressure_magnitudes(frame, [16, 16, 16])[0])
        frame['relative_pressure_difference'] = 0.01
        frame['pressure_comparison_scope'] = 'same_request'
        self.assertIsNone(pressure_magnitudes(frame, [16, 16, 16])[0])

    def test_enrichment_preserves_points_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frame = dict(step=16, time_s=0.25, relative_pressure_difference=0.01,
                         pressure_comparison_scope='independent_trajectories', ai_quality=dict(qualified=True))
            raw = json.dumps(frame).encode()
            (base/'frames.jsonl').write_bytes(raw)
            points = [[16, 0.25, 0.01, True, 1e-7]]
            trace = dict(frames_source='frames.jsonl', scene=dict(counts=[16]*3), points=points)
            data = dict(drift_traces=[trace],history_sources={'frames.jsonl':hashlib.sha256(raw).hexdigest()})
            enrich_drift_pressure(data, base)
            self.assertEqual(trace['points'], points)
            self.assertNotIn('pressure_magnitudes', trace)
            self.assertEqual(trace['pressure_magnitude_coverage'], {'same-request norms not recorded':1})
            (base/'frames.jsonl').write_bytes(raw+b'\n')
            with self.assertRaisesRegex(ValueError, 'Drift frame source changed'):
                enrich_drift_pressure(data, base)


if __name__ == '__main__':
    unittest.main()
