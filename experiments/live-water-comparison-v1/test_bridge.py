"""Boundary regressions; real CPU checkpoint parity is verified separately."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cpu_bridge
import deployment


class BridgeBoundaryTests(unittest.TestCase):
    def request(self, **updates):
        value = {
            'id': 16, 'counts_xyz': [5, 5, 5], 'cell_size_m': 0.0625,
            'density_kg_m3': 1000.0, 'dt_s': 1 / 60,
            'base_pressure': [0.0] * 125, 'native_input_residual': [0.0] * 125,
        }
        return {**value, **updates}

    def setUp(self):
        # Invalid requests must fail before accessing the numerical runtime.
        self.state = (None, None, None, None, None, {'minimum_native_steps': 15})

    def test_cold_start_is_unsupported_context_before_model_access(self):
        with self.assertRaises(cpu_bridge.ProtocolError) as caught:
            cpu_bridge.predict(self.request(id=1), self.state)
        self.assertEqual(caught.exception.code, 'unsupported_context')
        reply = cpu_bridge.failure(caught.exception, 1)
        self.assertEqual(reply['id'], 1)
        self.assertEqual(reply['code'], 'unsupported_context')

    def test_closed_request_shape_spacing_and_identity(self):
        cases = [
            self.request(extra=True), self.request(id=True),
            self.request(id=2**53), self.request(counts_xyz=[5, 5]),
            self.request(counts_xyz=[128, 128, 128]),
            self.request(cell_size_m=True), self.request(cell_size_m=float('nan')),
            self.request(density_kg_m3=1.25), self.request(dt_s=0.1),
        ]
        for value in cases:
            with self.subTest(value=list(value)):
                with self.assertRaises(ValueError):
                    cpu_bridge.predict(value, self.state)

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for raw in ('{"id":16,"id":17}', '{"value":NaN}', '{"value":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                cpu_bridge.strict(raw)
        self.assertEqual(cpu_bridge.strict('{"id":16}'), {'id': 16})


class DeploymentContextTests(unittest.TestCase):
    def test_context_is_derived_from_measured_training_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inputs').mkdir()
            path = root / 'inputs/splits.json'
            raw = json.dumps({'reservation': {'configuration': {'snapshot_steps': [3, 31]}}}).encode()
            path.write_bytes(raw)
            receipt = {'closure': {'inputs': {'splits': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}}}
            self.assertEqual(deployment.context_policy(root, receipt), {
                'mode': 'native_prefix', 'minimum_native_steps': 3, 'training_snapshot_indices': [3, 31]})
            path.write_bytes(raw.replace(b'[3, 31]', b'[0, 31]'))
            with self.assertRaisesRegex(ValueError, 'reservation drift'):
                deployment.context_policy(root, receipt)


if __name__ == '__main__':
    unittest.main()
