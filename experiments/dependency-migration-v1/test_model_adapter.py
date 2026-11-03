"""Current protocol rejection tests use no neural runtime or model execution."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import model_adapter as adapter
import package_model_adapter as package


def manifest():
    return {'context': {'minimum_native_steps': 15},
            'support': {'counts_min': [5]*3, 'counts_max': [128]*3, 'max_cells': 65536,
                        'spacing_min_m': 1/128, 'spacing_max_m': 1.},
            'inputs': {'base': 'previous_pressure', 'residual': 'initial_residual', 'rho': 'density', 'dt': 'dt'}}


def request():
    fields = {}
    for name, semantic in manifest()['inputs'].items():
        scalar = semantic in ('density', 'dt')
        fields[name] = {'semantic': semantic,
            'units': {'previous_pressure': 'Pa', 'initial_residual': 'm3/s2', 'density': 'kg/m3', 'dt': 's'}[semantic],
            'association': 'global' if scalar else 'cell', 'dimensions': [1]*3 if scalar else [5]*3,
            'scalar': 'f64', 'layout': 'x_fast',
            'values': [1000.] if semantic == 'density' else [1/60] if semantic == 'dt' else [0.]*125}
    return {'kind': 'predict', 'schema': 'rheon-pressure-context-request-v1', 'stage': adapter.STAGE, 'id': 16,
            'context': {'run_id': 'test', 'generation': 16, 'mode': 'diagnostic_candidate',
                'interval_start_s': 15/60, 'interval_end_s': 15/60+1/60, 'dt_s': 1/60,
                'grid': {'counts_xyz': [5]*3, 'spacing_xyz_m': [.0625]*3, 'origin_xyz_m': [0.]*3},
                'prior': {'generation': 15, 'interval_end_s': 15/60, 'status': {'kind': 'accepted'}}}, 'inputs': fields}


class RequestTests(unittest.TestCase):
    def assert_rejected(self, value, code):
        state = adapter.State(None, None, None, None, {}, {}, manifest())
        with self.assertRaises(adapter.ProtocolError) as caught:
            adapter.predict(value, state)
        self.assertEqual(caught.exception.code, code)

    def test_current_valid_envelope_preserves_order_and_values(self):
        value = request()
        counts, spacing, fields = adapter.validate_request(value, manifest())
        self.assertEqual((counts, spacing), ([5]*3, .0625))
        self.assertIs(fields['previous_pressure'], value['inputs']['base']['values'])

    def test_old_envelope_and_unknown_protocol_rejected_before_model_access(self):
        self.assert_rejected({'id': 16, 'counts_xyz': [5]*3, 'base_pressure': [0.]*125}, 'invalid')
        value = request()
        value['schema'] = 'rheon-pressure-context-request-v2'
        self.assert_rejected(value, 'unsupported')

    def test_native_prefix_solved_lineage_and_accepted_history(self):
        cases = []
        for edit in ('cold', 'missing', 'stale', 'unqualified', 'diagnostic_accepted'):
            value = request()
            context = value['context']
            if edit == 'cold':
                context['generation'] = 15
            elif edit == 'missing':
                context['prior'] = None
            elif edit == 'stale':
                context['prior']['generation'] = 14
            elif edit == 'unqualified':
                context['prior']['status'] = {'kind': 'diagnostic', 'qualified': False}
            else:
                context['mode'] = 'accepted_candidate'
                context['prior']['status'] = {'kind': 'diagnostic', 'qualified': True}
            cases.append(value)
        for value in cases:
            with self.subTest(context=value['context']):
                self.assert_rejected(value, 'unsupported_context')

    def test_physical_mismatches_rejected_before_model_access(self):
        for edit in ('density', 'spacing', 'origin', 'counts', 'dt'):
            value = request()
            grid = value['context']['grid']
            if edit == 'density': value['inputs']['rho']['values'] = [999.]
            elif edit == 'spacing': grid['spacing_xyz_m'][1] = .125
            elif edit == 'origin': grid['origin_xyz_m'][0] = 1.
            elif edit == 'counts': grid['counts_xyz'] = [128]*3
            else: value['context']['dt_s'] = .1
            with self.subTest(edit=edit):
                self.assert_rejected(value, 'unsupported')

    def test_indexed_clock_endpoints_through_step_512_keep_exact_prior_alignment(self):
        for step in range(16, 513):
            value = request()
            context = value['context']
            context.update(generation=step, interval_start_s=(step-1)*(1/60), interval_end_s=step*(1/60))
            context['prior'].update(generation=step-1, interval_end_s=(step-1)*(1/60))
            adapter.validate_request(value, manifest())
        value = request()
        value['context']['interval_end_s'] = math.nextafter(
            math.nextafter(value['context']['interval_end_s'], float('inf')), float('inf'))
        self.assert_rejected(value, 'unsupported')

    def test_field_types_units_shape_gauge_and_extra_fields(self):
        for edit in ('unit', 'scalar', 'length', 'nan', 'gauge', 'extra', 'bool_dimension', 'bool_value'):
            value = request()
            base = value['inputs']['base']
            if edit == 'unit': base['units'] = 'm'
            elif edit == 'scalar': base['scalar'] = 'f32'
            elif edit == 'length': base['values'].pop()
            elif edit == 'nan': base['values'][3] = float('nan')
            elif edit == 'gauge': base['values'][0] = 1.
            elif edit == 'bool_dimension': value['inputs']['rho']['dimensions'][0] = True
            elif edit == 'bool_value': base['values'][3] = True
            else: base['extra'] = True
            with self.subTest(edit=edit):
                self.assert_rejected(value, 'invalid')

    def test_strict_json_and_correlation_are_bounded(self):
        for raw in ('{"id":1,"id":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(adapter.ProtocolError):
                adapter.strict(raw)
        for identity in (True, -1, 2**53, '16'):
            value = request()
            value['id'] = identity
            self.assert_rejected(value, 'invalid')


class ConversionTests(unittest.TestCase):
    def test_archive_copies_exact_bytes_and_refuses_existing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original'
            original.write_bytes(b'original\x00\xff\n')
            archived = package.archive_files({'model/source.py': original}, root / 'archive')
            self.assertEqual((root / 'archive/model/source.py').read_bytes(), original.read_bytes())
            self.assertEqual(archived['model/source.py']['sha256'], adapter.measure(original)['sha256'])
            with self.assertRaises(FileExistsError):
                package.archive_files({'model/source.py': original}, root / 'archive')

    def test_policy_extraction_excludes_historical_cli(self):
        original = Path(__file__).resolve().parents[1] / 'water-stability-learning-v1/learned_iteration.py'
        extracted = package.iteration_functions(original)
        self.assertNotIn('def main', extracted)
        self.assertNotIn('read_view', extracted)
        self.assertIn('def correct(', extracted)
        compile(extracted, '<current-policy>', 'exec')


if __name__ == '__main__':
    unittest.main()
