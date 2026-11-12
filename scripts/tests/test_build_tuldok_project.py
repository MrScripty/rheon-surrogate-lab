"""Tests for scripts/build_tuldok_project.py: the conversion rules on small made-up projects.

    .venv/bin/python -m unittest discover -s scripts/tests -v

The end-to-end tests need Tuldok's tuldok_project package, found at $TULDOK_REPOSITORY or
beside this repository (../../creative-media/Tuldok); they are skipped without it. No test
reads the earlier project folder or compares projects/surrogate-lab with the converter's
output: that folder is edited by hand now.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / 'scripts'))

import build_tuldok_project as converter  # noqa: E402

TULDOK = Path(os.environ.get('TULDOK_REPOSITORY') or LAB.parents[1] / 'creative-media' / 'Tuldok')
HAS_TULDOK = (TULDOK / 'tuldok_project' / 'results.py').is_file()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding='utf-8')


TRAINING_SETS = {
    'about': 'test',
    'recordings': [
        {'id': 'sinusoidal-recordings', 'what': 'Scenes.', 'dataset': 'sinusoidal-v1'},
        {'id': 'sealed-water-capture', 'what': 'Water scenes.', 'dataset': None,
         'originally_captured_by': ['experiments/sealed-water-v1/pipeline.py'], 'notes': 'Not recorded.'},
    ],
    'training_sets': [
        {'id': 'temporal-training-view-02', 'what': 'Packed states.', 'prepared_by': ['a.py', 'b.py'],
         'command_recorded': False, 'command': 'a.py then b.py', 'built_from': ['sinusoidal-recordings (train split)']},
        {'id': 'internal-residual-view-01', 'what': 'Residuals.', 'prepared_by': ['c.py'], 'command_recorded': False,
         'command': 'c.py --protocol <settings file> --output <folder> (needs CUDA)',
         'built_from': ['temporal-training-view-02'], 'needs_models': ['some model']},
        {'id': 'pressure-methods-v2-shape-view', 'what': 'Shapes.', 'prepared_by': ['d.py'], 'command_recorded': True,
         'command': 'd.py --output runs/x', 'built_from': ['internal-residual-view-01']},
        {'id': 'variable-pressure-project-inputs', 'what': 'Variable.', 'prepared_by': ['e.py'],
         'command_recorded': False, 'command': None, 'built_from': ['sealed-water-capture']},
    ],
    'not_training_data': [{'id': 'environment-listing', 'what': 'Listing.'}],
}


def recorded_recipe(stem, model, fit, inputs, command='train --data <input: data> --config <source: config.json> '
                                                      '--evidence <input evidence> --output-dir <output dir>',
                    sources=('train.py', 'runtime-bootstrap/extension-binding.json')):
    staged = f'runs/study/{stem}'
    return {'model': model, 'fit_id': fit, 'card': f'notes/fits/{stem}.md', 'tuldok_run': 'r' * 32,
            'run_name': model, 'trained_at': '2026-10-15T10:25:40+00:00', 'device': 'cuda',
            'runtime': 'managed_torch_python_v1', 'resources': {'wall_seconds': 3600, 'max_output_bytes': 10 ** 9},
            'entrypoint': f'{staged}/train.py', 'command': command, 'parameters': {},
            'settings_file': f'{staged}/config.json', 'settings': {'seed': 1},
            'inputs': inputs, 'sources': [f'{staged}/config.json', *(f'{staged}/{name}' for name in sources)],
            'outputs': [{'path': 'weights.npz', 'role': 'checkpoint'}, {'path': 'history.json', 'role': None}]}


TRAIN_SCRIPT = b'print("train")\n'


def make_code(root, stems):
    """A Git repository at `root` tracking experiments/study/train.py, with staged copies of it under runs/."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'experiments' / 'study').mkdir(parents=True, exist_ok=True)
    (root / 'experiments' / 'study' / 'train.py').write_bytes(TRAIN_SCRIPT)
    (root / 'experiments' / 'other').mkdir(parents=True, exist_ok=True)
    (root / 'experiments' / 'other' / 'train.py').write_bytes(TRAIN_SCRIPT)
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'add', 'experiments'], cwd=root, check=True)
    for stem in stems:
        staged = root / 'runs' / 'study' / stem
        staged.mkdir(parents=True, exist_ok=True)
        (staged / 'train.py').write_bytes(TRAIN_SCRIPT)
        (staged / 'config.json').write_text('{"seed": 1}')
        (staged / 'local.py').write_text('# never committed\n')
        (staged / 'qualification.json').write_text('{"status": "smoke"}')
        (staged / 'source-witness.json').write_text('{"inputs": {}}')
        (staged / 'weights.npz').write_bytes(b'PK')
        (staged / 'unknown.json').write_text('{}')
        (staged / 'model-contribution.json').write_text('{"format": "tuldok_contribution_v1"}')


class EverythingTracked:
    """Stands in for CodePaths in tests that are not about code paths."""
    def __init__(self, untracked=()):
        self.untracked = set(untracked)

    def resolve(self, path):
        return ('missing', path) if path in self.untracked else ('tracked', path)


class Helpers(unittest.TestCase):
    def test_subject_names(self):
        self.assertEqual(converter.subject_name('C106'), 'c106')
        self.assertEqual(converter.subject_name('temporal_eager.fine_reference'), 'temporal_eager.fine_reference')
        self.assertEqual(converter.subject_name('F64 N1/K4'), 'f64-n1-k4')

    def test_keys(self):
        self.assertEqual(converter.key_of('checkpoint-000.npz'), 'checkpoint_000_npz')
        self.assertEqual(converter.key_of('127-models'), 'x_127_models')
        self.assertLessEqual(len(converter.key_of('a' * 50)), 32)

    def test_result_set_groups(self):
        group = converter.result_set_group
        self.assertEqual(group('C17–C32 · original pressure cases (16 subjects)'), 'original pressure cases')
        self.assertEqual(group('C130–C130 · original pressure cases (1 subjects)'), 'original pressure cases')
        self.assertEqual(group('150 pressure fits · complete GPU requests · 128 consumed cases · part 2'),
                         '150 pressure fits · complete GPU requests · 128 consumed cases')
        # Two different benchmarks that share the text after their first '·' stay apart.
        self.assertNotEqual(group('Temporal sampler · complete GPU blocks · frozen old physics'),
                            group('Cell sampler · complete GPU blocks · frozen old physics'))

    def test_dimension_text(self):
        self.assertEqual([converter.dimension_text(v) for v in (None, True, 3, 0.5, 'x')],
                         ['null', 'true', '3', '0.5', 'x'])
        with self.assertRaises(ValueError):
            converter.dimension_text('a=b')

    def test_owned_files(self):
        self.assertTrue(converter.owned_by_script('notes/fits/a.md', b''))
        self.assertTrue(converter.owned_by_script('recipes/a.json', b'{"earlier_work": {"rebuild": "yes"}}'))
        self.assertFalse(converter.owned_by_script('recipes/a.json', b'{"kind": "recipe"}'))
        self.assertFalse(converter.owned_by_script('README.md', b''))


class ValuesAndSummaries(unittest.TestCase):
    OBSERVED = [  # (case, metric, value, reason, dimension texts, result set)
        ('a', 'err', 1.0, None, {'blocks': '2', 'pass': 'true', 'samples': '20'}, 'set 1'),
        ('b', 'err', 3.0, None, {'blocks': '4', 'pass': 'true', 'samples': '20'}, 'set 1'),
        ('c', 'err', None, 'Not scored.', {'blocks': 'null', 'pass': 'false', 'samples': '20'}, 'set 2'),
    ]

    def test_values_keep_only_group_by_keys(self):
        values = converter.case_values(self.OBSERVED, ['blocks', 'pass'], 'test')
        self.assertEqual(values, [
            {'case': 'a', 'metric': 'err', 'value': 1.0, 'group': {'blocks': '2', 'pass': 'true'}},
            {'case': 'b', 'metric': 'err', 'value': 3.0, 'group': {'blocks': '4', 'pass': 'true'}},
            # A null dimension is not a group value.
            {'case': 'c', 'metric': 'err', 'value': None, 'reason': 'Not scored.', 'group': {'pass': 'false'}}])

    def test_marginal_rows_only(self):
        def count(values, unavailable):
            return {'count': len(values), 'unavailable': unavailable, 'total': sum(values)}
        values = converter.case_values(self.OBSERVED, ['blocks', 'pass'], 'test')
        rows = converter.summarise_values(values, count)
        self.assertEqual(rows, {
            ('err', ''): {'count': 2, 'unavailable': 1, 'total': 4.0},
            ('err', 'blocks=2'): {'count': 1, 'unavailable': 0, 'total': 1.0},
            ('err', 'blocks=4'): {'count': 1, 'unavailable': 0, 'total': 3.0},
            ('err', 'pass=false'): {'count': 0, 'unavailable': 1, 'total': 0},
            ('err', 'pass=true'): {'count': 2, 'unavailable': 0, 'total': 4.0}})

    def test_duplicate_value_is_refused(self):
        observed = self.OBSERVED + [('a', 'err', 1.5, None, {'blocks': '2', 'pass': 'true', 'samples': '20'}, 'set 3')]
        with self.assertRaises(ValueError) as raised:
            converter.case_values(observed, ['blocks', 'pass'], 'test')
        self.assertIn("'set 1'", str(raised.exception))
        self.assertIn("'set 3'", str(raised.exception))

    def test_values_file_is_strict_json(self):
        document = {'format': 'tuldok_results_v1', 'benchmark': 'b', 'subject': 's', 'commit': '',
                    'produced': '2026-10-24', 'values': converter.case_values(self.OBSERVED, ['pass'], 'test')}
        self.assertEqual(json.loads(converter.dump_values(document)), document)


class TrainingSetConversion(unittest.TestCase):
    def setUp(self):
        self.sets = converter.TrainingSets(TRAINING_SETS, EverythingTracked())
        self.files = self.sets.definitions()

    def test_recordings(self):
        self.assertNotIn('datasets/sinusoidal-recordings.json', self.files)
        self.assertEqual(self.files['datasets/sealed-water-capture.json']['earlier_work']['rebuild'], 'no')
        self.assertEqual(self.sets.reference('sinusoidal-recordings (train split)'), 'datasets/sinusoidal-v1')

    def test_labels(self):
        rebuild = {path: document['earlier_work']['rebuild'] for path, document in self.files.items()}
        self.assertEqual(rebuild['datasets/temporal-training-view-02.json'], 'yes')
        self.assertEqual(rebuild['steps/temporal-training-view-02-first.json'], 'yes')
        self.assertEqual(rebuild['datasets/internal-residual-view-01.json'], 'after-inputs')
        # The label is about the set alone: it reads a set made with weights, but needs none itself.
        self.assertEqual(rebuild['datasets/pressure-methods-v2-shape-view.json'], 'yes')
        self.assertEqual(rebuild['datasets/variable-pressure-project-inputs.json'], 'no')
        self.assertNotIn('run', self.files['datasets/variable-pressure-project-inputs.json'])

    def test_untracked_script_makes_it_no(self):
        script = 'experiments/full-ai-next-priorities-v1/pack_temporal.py'
        files = converter.TrainingSets(TRAINING_SETS, EverythingTracked([script])).definitions()
        label = files['datasets/temporal-training-view-02.json']['earlier_work']
        self.assertEqual(label['rebuild'], 'no')
        self.assertIn(script, label['reason'])

    def test_two_stages_become_a_step_and_a_dataset(self):
        dataset = self.files['datasets/temporal-training-view-02.json']
        self.assertEqual(dataset['run']['inputs'], {'first': 'steps/temporal-training-view-02-first',
                                                    'sinusoidal_v1': 'datasets/sinusoidal-v1'})
        step = self.files['steps/temporal-training-view-02-first.json']
        self.assertEqual(step['run']['command'][1], 'experiments/full-ai-next-priorities-v1/pack_temporal.py')

    def test_cuda_environment_from_the_account(self):
        run = self.files['datasets/internal-residual-view-01.json']['run']
        self.assertEqual(run['environment'], 'environments/cuda-torch')
        self.assertEqual(run['command'][-2:], ['--output', '{output}'])


class RecipeConversion(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root))
        self.folder = self.root / 'recipes'
        make_code(self.root / 'lab', ['parent--aaaa', 'child--bbbb', 'local--dddd', 'twin--eeee', 'data--ffff'])
        data = {'name': 'data', 'path': 'runs/p/inputs/data.npz', 'bytes': 1, 'kind': 'training data',
                'training_set': 'internal-residual-view-01'}
        write_json(self.folder / 'parent--aaaa.json', recorded_recipe('parent--aaaa', 'Parent model', 'fit-a', [
            {**data, 'training_set': 'temporal-training-view-02'}]))
        write_json(self.folder / 'child--bbbb.json', recorded_recipe(
            'child--bbbb', 'Child model', 'fit-b',
            [data, {'name': 'start', 'path': 'runs/p/weights.npz', 'bytes': 1, 'kind': 'earlier model weights',
                    'from_model': 'Parent model'},
             {'name': 'mystery', 'path': 'runs/q/{w}.npz', 'bytes': 1, 'kind': 'earlier model weights'}],
            command='train --data <input: data> --init <input: start> --other <input: mystery> --output-dir <output dir>'))
        write_json(self.folder / 'local--dddd.json', recorded_recipe(
            'local--dddd', 'Local model', 'fit-d', [data],
            command='train --data <input: data> --witness <source: source-witness.json> --output-dir <output dir>',
            sources=('train.py', 'local.py', 'model-contribution.json', 'qualification.json', 'source-witness.json')))
        write_json(self.folder / 'twin--eeee.json', recorded_recipe(
            'twin--eeee', 'Twin model', 'fit-e', [data], sources=('train.py', 'local.py')))
        write_json(self.folder / 'data--ffff.json', recorded_recipe(
            'data--ffff', 'Data model', 'fit-f', [data], sources=('train.py', 'weights.npz')))
        write_json(self.folder / 'old--cccc.json', {
            'model': 'old', 'fit_id': 'fit-c', 'card': 'notes/fits/old--cccc.md', 'tuldok_run': None,
            'run_folder': 'runs/x', 'note': 'Trained before runs were recorded.', 'entrypoint': 'train.py',
            'command': None, 'settings_file': 'config.json', 'settings': {}, 'inputs': [], 'sources': [],
            'outputs': []})
        self.code_paths = converter.CodePaths(self.root / 'lab', exclude='projects/old')
        self.recipes = converter.Recipes(self.folder, self.code_paths)
        self.code_paths.prepare(path for recipe in self.recipes.recorded.values() if recipe.get('command')
                                for path in [recipe['entrypoint'], *recipe['sources']]
                                if path != recipe['settings_file'])
        self.files = self.recipes.definitions()

    def test_recorded_command_becomes_placeholders(self):
        run = self.files['recipes/parent--aaaa.json']['run']
        self.assertEqual(run['command'], ['python', 'experiments/study/train.py', 'train', '--data', '{inputs.data}',
                                          '--config', '{settings}', '--evidence', '<input evidence>',
                                          '--output-dir', '{output}'])
        self.assertEqual(run['inputs'], {'data': 'datasets/temporal-training-view-02'})
        self.assertEqual(run['settings'], {'seed': 1})
        self.assertEqual(run['resources'], {'gpu': True, 'wall_hours': 1.0, 'disk_gb': 1.0})
        self.assertEqual(run['outputs']['history_json'], {'path': 'history.json', 'role': 'file'})
        self.assertEqual(self.recipes.unknown_tokens['<input evidence>'], 3)

    def test_labels_and_card(self):
        document = self.files['recipes/parent--aaaa.json']
        self.assertEqual(document['title'], 'Parent model')
        self.assertEqual(document['card'], 'notes/fits/parent--aaaa.md')
        self.assertEqual(document['labels']['fit_id'], 'fit-a')
        self.assertEqual(document['labels']['input:data'], 'runs/p/inputs/data.npz')
        self.assertEqual(document['earlier_work']['rebuild'], 'needs-work')
        self.assertIn('<input evidence>', document['earlier_work']['reason'])

    def test_code_paths_become_tracked_files(self):
        run = self.files['recipes/parent--aaaa.json']['run']
        # The settings file is carried as settings, the generated binding is left out, and the staged
        # train.py becomes its tracked twin in the run's own study rather than experiments/other/.
        self.assertEqual(run['sources'], ['experiments/study/train.py'])
        self.assertEqual(self.code_paths.counts['generated'], 3)
        self.assertEqual(self.code_paths.counts['carried as settings'], 5)

    def test_untracked_code_is_copied_once(self):
        home = 'earlier-run-code/study/local--dddd/local.py'
        self.assertEqual(sorted(self.code_paths.copies), [home])
        self.assertEqual(self.code_paths.copies[home], b'# never committed\n')
        self.assertIn(home, self.files['recipes/local--dddd.json']['run']['sources'])
        self.assertIn(home, self.files['recipes/twin--eeee.json']['run']['sources'])
        self.assertEqual(self.files['recipes/twin--eeee.json']['earlier_work']['rebuild'], 'needs-work')

    def test_record_files_are_left_out_and_named_when_read(self):
        document = self.files['recipes/local--dddd.json']
        sources = document['run']['sources']
        self.assertFalse([s for s in sources if s.endswith(('qualification.json', 'source-witness.json',
                                                             'model-contribution.json'))])
        self.assertEqual(document['earlier_work']['rebuild'], 'needs-work')
        self.assertIn('<source: source-witness.json> (a record file', document['earlier_work']['reason'])
        self.assertIn('<source: source-witness.json>', document['run']['command'])

    def test_only_arrays_missing_is_no(self):
        label = self.files['recipes/data--ffff.json']['earlier_work']
        self.assertEqual(label['rebuild'], 'no')
        self.assertIn('only missing pieces are archives or arrays', label['reason'])
        self.assertIn('runs/study/data--ffff/weights.npz', label['reason'])

    def test_unclassified_json_is_refused(self):
        with self.assertRaises(ValueError):
            converter.CodePaths.file_kind('runs/study/x/unknown.json')
        self.assertEqual(converter.CodePaths.file_kind('runs/a/qualification.json'), 'record')
        self.assertEqual(converter.CodePaths.file_kind('runs/a/fine-tuning.json'), 'settings')

    def test_earlier_weights(self):
        document = self.files['recipes/child--bbbb.json']
        self.assertEqual(document['earlier_work']['rebuild'], 'no')
        self.assertIn('runs/q/{w}.npz', document['earlier_work']['reason'])
        self.assertEqual(document['run']['inputs']['start'], 'recipes/parent--aaaa')
        self.assertNotIn('mystery', document['run']['inputs'])
        self.assertIn('runs/q/{{w}}.npz', document['run']['command'])
        self.assertIn('does not tie to a model', document['description'])

    def test_no_command(self):
        document = self.files['recipes/old--cccc.json']
        self.assertEqual(document['earlier_work']['rebuild'], 'no')
        self.assertNotIn('run', document)


def make_source(root):
    """A tiny earlier project: one card, one recipe, two result sets of one benchmark."""
    notes = root / 'notes'
    (notes / 'fits').mkdir(parents=True)
    (notes / 'fits' / 'model--aaaa.md').write_text('# Model\n')
    (notes / 'overview.md').write_text('Overview\n')
    write_json(notes / 'recipes' / 'training-sets.json', TRAINING_SETS)
    write_json(notes / 'recipes' / 'model--aaaa.json', recorded_recipe('model--aaaa', 'Model', 'fit-a', [
        {'name': 'data', 'path': 'runs/p/data.npz', 'bytes': 1, 'kind': 'training data',
         'training_set': 'temporal-training-view-02'}],
        sources=('train.py', 'local.py', 'runtime-bootstrap/extension-binding.json')))
    write_json(notes / 'portfolio-index.json', {
        'aliases': [{'label': 'C02', 'fit_item_id': 'fit-a', 'declared_weights_sha256': 'w2',
                     'fit_link_disposition': 'linked_by_project_or_pair'}],
        'fits': [], 'source_candidates': [], 'limitations': []})
    write_json(notes / '00-models.portfolio.json', {
        'models': [{'id': 'fit-a', 'group_id': 'pressure', 'label': 'Model', 'aliases': []}],
        'groups': [{'id': 'pressure', 'label': 'Pressure fits', 'parent': None, 'description': 'Grouping.'}],
        'relationships': [{'kind': 'comparison', 'from': 'fit-a', 'to': 'fit-z', 'evidence_model': 'fit-a',
                           'explanation': 'A compared with Z.', 'excerpt': 'Better.'}]})
    write_json(notes / 'checkpoints' / 'part-01.json', {'rows': []})
    inventory = []
    for part, subjects in ((1, [{'id': 'C01', 'label': 'C01', 'dimensions': {'fit_item_id': 'fit-a'}}]),
                           (2, [{'id': 'C02', 'label': 'C02', 'dimensions': {'weights_sha256': 'w2'}},
                                {'id': 'F64-N1', 'label': 'F64', 'dimensions': {'origin_kind': 'assembly_execution'}}])):
        observations = [{'subject_id': s['id'], 'case_id': case, 'metric_id': 'error-l2', 'state': 'measured',
                         'value': value, 'dimensions': {'blocks': value > 1}}
                        for s in subjects for case, value in (('a', 1.0), ('b', 3.0))]
        observations.append({'subject_id': subjects[0]['id'], 'case_id': 'c', 'metric_id': 'error-l2',
                             'state': 'unavailable', 'value': None, 'dimensions': [{'id': 'blocks', 'value': False}]})
        sha = f'{part:064x}'
        write_json(root / 'artifacts' / 'sha256' / sha[:2] / sha, {
            'title': 'x', 'scope': {'id': 'scope-1', 'label': 'Scope', 'population': 'consumed'},
            'metrics': [{'id': 'error-l2', 'label': 'Error', 'units': '1'}], 'dimensions': [],
            'cases': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}], 'subjects': subjects, 'observations': observations,
            'sources': []})
        record = f'records/measurement_source/{part}.json'
        write_json(root / '.tuldok' / record, {'title': f'Errors · part {part}', 'blob': {'sha256': sha},
                                                'imported_at': '2026-10-24T06:20:32+00:00'})
        inventory.append({'family': 'measurement_source', 'path': record})
    write_json(root / '.tuldok' / 'project.json', {'inventory': inventory})


def store(source, data):
    """Put `data` in the old folder's artifacts/sha256/ under its checksum; return the checksum."""
    checksum = hashlib.sha256(data).hexdigest()
    path = source / 'artifacts' / 'sha256' / checksum
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return checksum


def write_run(source, run_id, outputs, payloads=(), stored=None):
    """An old run_history record with these outputs ({name: bytes}) and other payloads ((name, role, bytes)).
    Every file is put in artifacts/sha256/ except outputs not named in `stored` (when given)."""
    artifacts, retained = [], []
    for name, data in outputs.items():
        checksum = store(source, data) if stored is None or name in stored else hashlib.sha256(data).hexdigest()
        artifacts.append({'name': name, 'role': 'checkpoint', 'bytes': len(data), 'sha256': checksum,
                          'media_type': 'application/octet-stream'})
        retained.append({'name': name, 'role': 'checkpoint', 'blob': {'bytes': len(data), 'sha256': checksum}})
    for name, role, data in payloads:
        retained.append({'name': name, 'role': role, 'blob': {'bytes': len(data), 'sha256': store(source, data)}})
    write_json(source / '.tuldok' / 'records' / 'run_history' / f'catalog.{run_id}' / 'record.json', {
        'format': 'project_run_history_v2', 'payloads': retained,
        'run': {'id': run_id, 'state': 'completed', 'created_at': '2026-10-21T08:00:00+00:00', 'artifacts': artifacts}})


def recipe_definition(run_id, outputs):
    return converter.dump(converter.definition(
        'recipe', 'r', labels={'tuldok_run': run_id} if run_id else None, earlier_work={'rebuild': 'no'},
        run={'command': ['x'], 'outputs': converter.Recipes.outputs([{'path': p, 'role': 'checkpoint'} for p in outputs])}
        if outputs else None))


class Adoption(unittest.TestCase):
    def setUp(self):
        self.source = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(self.source))
        source = self.source
        write_run(source, 'a' * 32, {'weights.npz': b'A weights', 'history.json': b'A history'},
                  payloads=[('input/data', 'execution_source', b'training data'),
                            ('source/train.py', 'execution_source', b'print()'),
                            ('custody-receipt.json', 'custody_proof', b'{}')])
        write_run(source, 'b' * 32, {'weights.npz': b'B weights', 'history.json': b'B history', 'extra.npy': b'x'},
                  stored={'weights.npz', 'extra.npy'})
        write_run(source, 'c' * 32, {'weights.npz': b'C weights'}, stored=set())
        write_run(source, 'd' * 32, {'probe.npy': b'probe'}, payloads=[('input/x', 'execution_source', b'other input')])
        # A stored file of the wrong size, one only another record names, and one no record names.
        write_run(source, 'e' * 32, {'weights.npz': b'E weights'})
        (source / 'artifacts' / 'sha256' / hashlib.sha256(b'E weights').hexdigest()).write_bytes(b'E weights, longer')
        write_json(source / '.tuldok' / 'records' / 'model_material' / 'm.json',
                   {'artifacts': [{'blob': {'sha256': store(source, b'model material')}}]})
        store(source, b'nobody')
        # Runs that read datasets: the same file through different input names, a path of one dataset that two
        # different files claim, a file outside its dataset's folders, and datasets of several folders.
        write_run(source, 'f' * 32, {}, payloads=[('input/train', 'execution_source', b'training data'),
                                                  ('input/labels', 'execution_source', b'labels'),
                                                  ('input/v1', 'execution_source', b'view one')])
        write_run(source, 'g' * 32, {}, payloads=[('input/lab', 'execution_source', b'labels'),
                                                  ('input/data', 'execution_source', b'regenerated'),
                                                  ('input/stray', 'execution_source', b'stray'),
                                                  ('input/w1', 'execution_source', b'w one'),
                                                  ('input/w2', 'execution_source', b'w two')])
        dataset = converter.dump(converter.definition('dataset', 'd', earlier_work={'rebuild': 'no'}))
        self.files = {
            'recipes/a.json': recipe_definition('a' * 32, ['weights.npz', 'history.json']),
            'recipes/b.json': recipe_definition('b' * 32, ['weights.npz', 'history.json']),
            'recipes/c.json': recipe_definition('c' * 32, ['weights.npz']),
            'recipes/e.json': recipe_definition('e' * 32, ['weights.npz']),
            'recipes/old.json': recipe_definition(None, []),
            'datasets/set.json': dataset, 'datasets/multi.json': dataset, 'datasets/water.json': dataset,
            'datasets/nobody.json': dataset,
            'benchmarks/x.json': converter.dump(converter.definition('benchmark', 'x', earlier_work={'rebuild': 'no'})),
        }

        def read(name, path, training_set):
            return {'name': name, 'path': path, 'kind': 'training data', 'training_set': training_set}
        self.recorded = {
            'a': {'tuldok_run': 'a' * 32, 'inputs': [read('data', 'runs/s/set-01/data.npz', 'set')]},
            'e': {'tuldok_run': 'e' * 32, 'inputs': []}, 'old': {'tuldok_run': None, 'inputs': []},
            'f': {'tuldok_run': 'f' * 32, 'inputs': [read('train', 'runs/s/set-01/data.npz', 'set'),
                                                     read('labels', 'runs/s/set-01/sub/labels.json', 'set'),
                                                     read('v1', 'runs/a/v1/d.npz', 'multi')]},
            'g': {'tuldok_run': 'g' * 32, 'inputs': [read('lab', 'runs/s/set-01/sub/labels.json', 'set'),
                                                     read('data', 'runs/s/set-01/data.npz', 'set'),
                                                     read('stray', 'runs/elsewhere/x.npz', 'set'),
                                                     read('w1', 'runs/w-one/project/inputs/d.npz', 'water'),
                                                     read('w2', 'runs/w-two/project/inputs/d.npz', 'water')]}}
        self.sets = {'set': {'folders': ['runs/s/set-01']}, 'multi': {'folders': ['runs/a/v1', 'runs/b/v2']},
                     'water': {'folder_pattern': '^runs/w-[a-z]+/project/inputs$'}, 'nobody': {'folders': ['runs/n']}}
        self.before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        self.manifest, lines = converter.adoption(source, self.files, self.recorded, self.sets)
        self.printed = '\n'.join(lines)
        self.products = {entry['reference']: entry for entry in self.manifest['products']}

    def test_entries_key_present_files_by_declared_paths(self):
        products = self.products
        self.assertEqual(sorted(p for p in products if p.startswith('recipes/')), ['recipes/a', 'recipes/b'])
        a = products['recipes/a']
        self.assertEqual(a['note'], f"Tuldok run {'a' * 32} of 2026-10-21")
        self.assertEqual(sorted(a['files']), ['history.json', 'weights.npz'])
        for path in a['files'].values():
            self.assertTrue(Path(path).is_absolute())
            self.assertFalse(Path(path).is_symlink())
        self.assertEqual(Path(a['files']['weights.npz']).read_bytes(), b'A weights')
        # Only what is stored, and never an artifact the definition does not declare.
        self.assertEqual(list(products['recipes/b']['files']), ['weights.npz'])
        self.assertEqual(json.loads(converter.dump(self.manifest)), self.manifest)

    def test_dataset_entries_key_read_files_by_their_path_in_the_dataset(self):
        products = self.products
        self.assertEqual(sorted(p for p in products if p.startswith('datasets/')),
                         ['datasets/multi', 'datasets/set', 'datasets/water'])
        # Read by two runs through different input names; data.npz is claimed by two different files and the
        # stray file is outside the dataset's folder, so both are left out.
        dataset = products['datasets/set']
        self.assertEqual(dataset['note'], 'The files that 3 earlier Tuldok runs read from this dataset.')
        self.assertEqual(list(dataset['files']), ['sub/labels.json'])
        self.assertEqual(Path(dataset['files']['sub/labels.json']).read_bytes(), b'labels')
        # Several folders: keys are relative to the directory that holds them all.
        self.assertEqual(list(products['datasets/multi']['files']), ['a/v1/d.npz'])
        self.assertEqual(products['datasets/multi']['note'],
                         'The files that 1 earlier Tuldok run read from this dataset.')
        self.assertEqual(sorted(products['datasets/water']['files']),
                         ['w-one/project/inputs/d.npz', 'w-two/project/inputs/d.npz'])

    def test_dataset_placement(self):
        entry = {'folders': ['', 'project-01']}
        self.assertEqual(converter.dataset_key(entry, 'project-01/m.json', entry['folders']), 'project-01/m.json')
        self.assertEqual(converter.dataset_key({'folders': ['runs/s']}, 'runs/s/a/b.npz', ['runs/s']), 'a/b.npz')
        self.assertIsNone(converter.dataset_key({'folders': ['runs/s']}, 'runs/s/../t.npz', ['runs/s']))
        self.assertIsNone(converter.dataset_key({'folders': ['runs/s']}, 'runs/sx/t.npz', ['runs/s']))
        self.assertIsNone(converter.dataset_key({'folders': []}, 'runs/s/t.npz', []))

    def test_summary_counts_and_mismatches(self):
        printed = self.printed
        self.assertIn('7 files, 51 bytes', printed)
        self.assertIn('Earlier recipes: 5; every declared output has a file 1, some 1, none 2, no declared outputs 1. '
                      'Their entries name 3 files', printed)
        self.assertIn('Earlier datasets: 4, none declaring outputs; with an entry 3, without 1 (read by no earlier '
                      'recipe 1, every read file left out 0, more than 200 files 0). Their entries name 4 files',
                      printed)
        self.assertIn('read file that cannot be placed in its dataset: 1 reads (for example datasets/set: '
                      'runs/elsewhere/x.npz); 1 stored files', printed)
        self.assertIn('two or more different stored files claim, left out: 1 paths, 2 stored files, 0.000 GB '
                      '(for example datasets/set: data.npz (2 files))', printed)
        self.assertIn('old artifact with no declared output: 1 (for example recipes/b: extra.npy)', printed)
        self.assertIn('old artifact not in artifacts/sha256: 2', printed)
        self.assertIn('stored file not a regular file of the recorded size: 1 (for example recipes/e: weights.npz)',
                      printed)
        self.assertIn('declared output with no old artifact: 0', printed)

    def test_unnamed_files_by_what_refers_to_them(self):
        printed = self.printed
        self.assertIn('Files in artifacts/sha256 that no entry names: 11,', printed)
        for label in ("input of an earlier recipe's run: training data", 'code staged for an old run (source/)',
                      'custody_proof file of an old run', 'output (checkpoint) of an old run with a recipe, completed',
                      'output (checkpoint) of an old run with no recipe, completed',
                      'input of an old run with no recipe', 'named in model_material records', 'named by no record'):
            self.assertIn(f'  {label}', printed)

    def test_old_folder_is_unchanged(self):
        self.assertEqual({p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}, self.before)

    def test_linked_store_file_is_not_adopted(self):
        weights = self.source / 'artifacts' / 'sha256' / hashlib.sha256(b'A weights').hexdigest()
        moved = self.source / 'elsewhere'
        weights.rename(moved)
        weights.symlink_to(moved)
        manifest, lines = converter.adoption(self.source, self.files, self.recorded, self.sets)
        a, = [entry for entry in manifest['products'] if entry['reference'] == 'recipes/a']
        self.assertEqual(list(a['files']), ['history.json'])
        self.assertIn('stored file not a regular file of the recorded size: 2', '\n'.join(lines))


@unittest.skipUnless(HAS_TULDOK, f'Tuldok is not at {TULDOK}')
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root))
        self.source = self.root / 'projects' / 'old'
        make_source(self.source)
        make_code(self.root, ['model--aaaa'])
        self.output = self.root / 'projects' / 'surrogate-lab'
        write_json(self.output / 'models' / 'hand.json', {'format': 'tuldok_definition_v1', 'kind': 'model',
                                                          'title': 'Hand written'})

    def run_script(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            status = converter.main(['--tuldok', str(TULDOK), '--source', str(self.source),
                                     '--output', str(self.output), *extra])
        return status, printed.getvalue()

    def snapshot(self):
        return {p.relative_to(self.output).as_posix(): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}

    def test_writes_deterministically_and_checks(self):
        self.assertEqual(self.run_script()[0], 0)
        first = self.snapshot()
        self.assertEqual(self.run_script()[0], 0)
        self.assertEqual(self.snapshot(), first)
        self.assertEqual(self.run_script('--check')[0], 0)
        (self.output / 'notes' / 'overview.md').write_text('changed\n')
        self.assertEqual(self.run_script('--check')[0], 1)
        self.assertEqual((self.output / 'notes' / 'overview.md').read_text(), 'changed\n')

    def test_folder_contents(self):
        status, printed = self.run_script()
        files = self.snapshot()
        self.assertIn('models/hand.json', files)
        self.assertIn('notes/fits/model--aaaa.md', files)
        self.assertIn('notes/overview.md', files)
        self.assertNotIn('notes/portfolio-index.json', files)
        self.assertEqual(json.loads(files['tuldok.json'])['code_root'], '../..')
        benchmark = json.loads(files['benchmarks/errors.json'])
        self.assertEqual(benchmark['metrics'], [{'id': 'error_l2', 'title': 'Error', 'units': '1', 'better': 'lower'}])
        self.assertEqual(benchmark['summary'], {'group_by': ['blocks']})
        self.assertEqual(benchmark['cases'], {'count': 3})
        recipes = {s['name']: s.get('recipe') for s in benchmark['subjects']}
        self.assertEqual(recipes, {'c01': 'recipes/model--aaaa', 'c02': 'recipes/model--aaaa', 'f64-n1': None})
        self.assertIn('f64-n1 (a solver assembly, not a trained model)', printed.replace(': F64-N1', ': f64-n1'))
        lines = files['results/errors.csv'].decode().splitlines()
        self.assertEqual(lines[0], 'subject,metric,group,count,unavailable,mean,median,p10,p90,min,max,commit,produced')
        self.assertIn('c01,error_l2,,2,1,2.0,2.0,1.2,2.8,1.0,3.0,,2026-10-24', lines)
        self.assertIn('c01,error_l2,blocks=false,1,1,1.0,1.0,1.0,1.0,1.0,1.0,,2026-10-24', lines)

    def test_earlier_run_code(self):
        self.run_script()
        code = self.root / 'earlier-run-code'
        copy = code / 'study' / 'model--aaaa' / 'local.py'
        self.assertEqual(copy.read_text(), '# never committed\n')
        self.assertIn('scripts/build_tuldok_project.py', (code / 'README.md').read_text())
        recipe = json.loads((self.output / 'recipes' / 'model--aaaa.json').read_text())
        self.assertIn('earlier-run-code/study/model--aaaa/local.py', recipe['run']['sources'])
        # The script owns the folder: --check sees an edit, and a stale file is removed.
        copy.write_text('edited\n')
        self.assertEqual(self.run_script('--check')[0], 1)
        (code / 'stale.py').write_text('x\n')
        self.run_script()
        self.assertFalse((code / 'stale.py').exists())
        self.assertEqual(copy.read_text(), '# never committed\n')
        self.assertEqual(self.run_script('--check')[0], 0)

    def test_groups_and_comparisons(self):
        self.run_script()
        document = json.loads((self.output / 'notes' / 'model-groups-and-comparisons.json').read_text())
        self.assertEqual(document['groups'], [{'id': 'pressure', 'title': 'Pressure fits', 'parent': None,
                                               'description': 'Grouping.', 'models': ['recipes/model--aaaa']}])
        self.assertEqual(document['comparisons'], [{'model': 'recipes/model--aaaa', 'compared_with': 'fit-z',
                                                    'explanation': 'A compared with Z.', 'excerpt': 'Better.'}])

    def test_values_files_rebuild_the_summaries(self):
        sys.path.insert(0, str(TULDOK))
        from tuldok_project.results import SummaryRow, summarise, write_results
        values = self.root / 'values'
        status, printed = self.run_script('--values-output', str(values))
        self.assertIn('Values files', printed)
        files = sorted(p.relative_to(values).as_posix() for p in values.rglob('*.json'))
        self.assertEqual(files, ['errors/c01.json', 'errors/c02.json', 'errors/f64-n1.json'])
        first = {path: (values / path).read_bytes() for path in files}
        rows = []
        for path in files:
            document = json.loads(first[path])
            for (metric, group), summary in converter.summarise_values(document['values'], summarise).items():
                rows.append(SummaryRow(subject=document['subject'], metric=metric, group=group,
                                       commit=document['commit'], produced=document['produced'], **summary))
        self.assertEqual(write_results(rows), (self.output / 'results' / 'errors.csv').read_text())
        c01 = json.loads(first['errors/c01.json'])
        self.assertIn({'case': 'c', 'metric': 'error_l2', 'value': None,
                       'reason': 'Recorded as unavailable by the earlier study harness.',
                       'group': {'blocks': 'false'}}, c01['values'])
        self.run_script('--values-output', str(values))
        self.assertEqual({path: (values / path).read_bytes() for path in files}, first)

    def test_duplicate_values_across_result_sets_stop_the_script(self):
        record = self.source / '.tuldok' / 'records' / 'measurement_source' / '2.json'
        sha = json.loads(record.read_text())['blob']['sha256']
        blob = self.source / 'artifacts' / 'sha256' / sha[:2] / sha
        document = json.loads(blob.read_text())
        document['subjects'].append({'id': 'C01', 'label': 'C01', 'dimensions': {'fit_item_id': 'fit-a'}})
        document['observations'].append({'subject_id': 'C01', 'case_id': 'a', 'metric_id': 'error-l2',
                                         'state': 'measured', 'value': 9.0, 'dimensions': {'blocks': False}})
        blob.write_text(json.dumps(document))
        with self.assertRaises(ValueError) as raised:
            self.run_script()
        self.assertIn('Errors · part 1', str(raised.exception))
        self.assertIn('Errors · part 2', str(raised.exception))

    def test_adoption_manifest(self):
        write_run(self.source, 'r' * 32, {'weights.npz': b'weights', 'history.json': b'{}'})
        before = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        manifest = self.root / 'adoption' / 'manifest.json'
        status, printed = self.run_script('--check', '--adoption-output', str(manifest))
        self.assertEqual(status, 1)    # the folder was never written; the manifest is written regardless
        self.assertIn('Earlier recipes: 1; every declared output has a file 1', printed)
        document = json.loads(manifest.read_text())
        self.assertEqual(document['format'], 'tuldok_adoption_v1')
        (entry,) = document['products']
        self.assertEqual(entry['reference'], 'recipes/model--aaaa')
        self.assertEqual(sorted(entry['files']), ['history.json', 'weights.npz'])
        self.run_script()
        recipe = json.loads((self.output / 'recipes' / 'model--aaaa.json').read_text())
        self.assertEqual(sorted(output['path'] for output in recipe['run']['outputs'].values()), sorted(entry['files']))
        self.assertEqual({p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}, before)

    def test_stale_owned_files_go_and_hand_written_stay(self):
        self.run_script()
        write_json(self.output / 'recipes' / 'gone.json', {'earlier_work': {'rebuild': 'no', 'reason': 'x'}})
        self.run_script()
        files = self.snapshot()
        self.assertNotIn('recipes/gone.json', files)
        self.assertIn('models/hand.json', files)

    def test_refuses_to_overwrite_hand_written(self):
        write_json(self.output / 'recipes' / 'model--aaaa.json', {'kind': 'recipe'})
        with self.assertRaises(ValueError):
            self.run_script()


if __name__ == '__main__':
    unittest.main()
