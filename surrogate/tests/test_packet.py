import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from surrogate.bench import packet, results

TULDOK = os.environ.get('TULDOK_CONTRACT_ROOT')

# Run by Tuldok's own interpreter path: its validator must accept every packet, and
# packets cut from one result must be comparable together.
CONSUMER = '''
import json, sys
from retained_measurements import MeasurementTable, MeasurementComparison
tables = [MeasurementTable(open(path, 'rb').read()) for path in sys.argv[1:]]
summary = {'subjects': sum(t.metadata()['subject_count'] for t in tables),
           'observations': sum(t.metadata()['observation_count'] for t in tables),
           'population': tables[0].metadata()['scope']['population']}
rows = tables[0].query('velocity_error_worst', limit=100)['rows']
summary['first_value'] = rows[0]['value']
summary['first_grid'] = rows[0]['dimensions']['grid_x']
if len(tables) > 1:
    located = [({'origin_catalog_id': 'a' * 32, 'packet_sha256': t.sha256}, t) for t in tables]
    summary['compared'] = MeasurementComparison(located).read('velocity_error_worst')['query']['total']
print(json.dumps(summary))
'''


def document(subjects=3):
    metrics = [
        {'id': 'velocity_error_worst', 'label': 'Worst velocity error', 'units': 'ratio', 'description': 'x'},
        {'id': 'pressure_error_worst', 'label': 'Worst pressure error', 'units': 'ratio', 'description': 'x'},
        {'id': 'time_per_output_ms', 'label': 'Time per output', 'units': 'ms', 'description': 'x'}]
    cases = [{'id': 'scene-a.start-0', 'scene': 'scene-a', 'grid_x': 16, 'cells_per_metre': 8},
             {'id': 'scene-b.start-24', 'scene': 'scene-b', 'grid_x': 32, 'cells_per_metre': 12.5}]
    names = [{'id': f'model-{index}', 'label': f'Model {index}', 'kind': 'learned', 'outputs_per_pass': 1 + index}
             for index in range(subjects)]
    rows = []
    for s, subject in enumerate(names):
        for c, case in enumerate(cases):
            values = {'velocity_error_worst': 0.1 / (1 + s + c), 'pressure_error_worst': 1e-3 * (1 + s),
                      'time_per_output_ms': 2}
            failure = None
            if (s, c) == (1, 1):
                values, failure = dict.fromkeys(values), 'Diverged at output 12:\n  velocity was not finite.'
            if s == 2:
                values['pressure_error_worst'] = None   # this subject predicts no pressure
            rows.append({'subject': subject['id'], 'case': case['id'], 'values': values, 'failure': failure})
    return {'format': results.FORMAT, 'git_commit': '0' * 40, 'git_dirty': False, 'environment': {},
            'suite': {'id': 'unit-suite', 'label': 'Unit test suite', 'population': 'independent',
                      'description': 'Synthetic.'},
            'metrics': metrics, 'cases': cases, 'subjects': names, 'results': rows}


class ResultDocument(unittest.TestCase):
    def test_round_trips_through_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = results.write(document(), Path(directory) / 'nested' / 'result.json')
            self.assertEqual(results.read(path), document())

    def test_rejects_incomplete_or_invalid_documents(self):
        missing = document()
        missing['results'].pop()
        with self.assertRaisesRegex(ValueError, 'have no result'):
            results.validate(missing)
        nonfinite = document()
        nonfinite['results'][0]['values']['time_per_output_ms'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'record a failure instead'):
            results.validate(nonfinite)
        bad_id = document()
        bad_id['subjects'][0]['id'] = 'has a space'
        bad_id['results'][0]['subject'] = bad_id['results'][1]['subject'] = 'has a space'
        with self.assertRaisesRegex(ValueError, 'letters, digits'):
            results.validate(bad_id)

    def test_git_identity_reports_this_repository(self):
        identity = results.git_identity()
        self.assertRegex(identity['git_commit'], r'^[0-9a-f]{40}$')
        self.assertIn(identity['git_dirty'], (True, False))


class Packets(unittest.TestCase):
    def test_every_value_points_at_the_same_number_in_the_embedded_result(self):
        (raw,) = packet.build(document())
        built = json.loads(raw)
        source = json.loads(base64.b64decode(built['sources'][0]['data_base64']))
        self.assertEqual(len(built['observations']), 3 * 2 * 3)
        states = {}
        for observation in built['observations']:
            states[observation['state']] = states.get(observation['state'], 0) + 1
            if observation['state'] != 'measured':
                self.assertIsNone(observation['value'])
                self.assertNotIn('\n', observation['reason'])
                continue
            value = source
            for token in observation['reference']['pointer'].split('/')[1:]:
                value = value[int(token)] if isinstance(value, list) else value[token]
            self.assertEqual(value, observation['value'])
        self.assertEqual(states, {'measured': 13, 'failed': 3, 'unsupported': 2})

    def test_mixed_integer_and_fractional_attributes_become_one_number_column(self):
        built = json.loads(packet.build(document())[0])
        column = next(d for d in built['dimensions'] if d['id'] == 'cells_per_metre')
        self.assertEqual((column['type'], column['scope']), ('number', 'case'))
        self.assertEqual(built['cases'][0]['dimensions']['cells_per_metre'], 8.0)

    def test_splits_by_subject_when_a_limit_is_reached(self):
        original = packet.MAX_ROWS
        packet.MAX_ROWS = 2 * 3 * 2   # two subjects per packet
        try:
            packets = [json.loads(raw) for raw in packet.build(document(subjects=5))]
        finally:
            packet.MAX_ROWS = original
        self.assertEqual([len(p['subjects']) for p in packets], [2, 2, 1])
        self.assertEqual(packets[0]['title'], 'Unit test suite · part 1 of 3')
        self.assertEqual({json.dumps(p['cases'], sort_keys=True) for p in packets}.__len__(), 1)

    @unittest.skipUnless(TULDOK, 'set TULDOK_CONTRACT_ROOT to a Tuldok checkout to run its validator')
    def test_tuldok_accepts_and_compares_the_packets(self):
        original = packet.MAX_ROWS
        packet.MAX_ROWS = 2 * 3 * 2
        try:
            packets = packet.build(document(subjects=5))
        finally:
            packet.MAX_ROWS = original
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, raw in enumerate(packets):
                paths.append(str(Path(directory) / f'packet-{index}.json'))
                Path(paths[-1]).write_bytes(raw)
            done = subprocess.run([sys.executable, '-c', CONSUMER, *paths], capture_output=True, text=True,
                                  env={**os.environ, 'PYTHONPATH': TULDOK, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(done.returncode, 0, done.stderr)
        summary = json.loads(done.stdout)
        self.assertEqual(summary, {'subjects': 5, 'observations': 30, 'population': 'independent',
                                   'first_value': 0.1, 'first_grid': 16, 'compared': 10})


if __name__ == '__main__':
    unittest.main()
