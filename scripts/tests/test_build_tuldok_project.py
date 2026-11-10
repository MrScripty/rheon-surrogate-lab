"""Tests for scripts/build_tuldok_project.py: the conversion rules on small made-up projects.

    .venv/bin/python -m unittest discover -s scripts/tests -v

The end-to-end tests need Tuldok's tuldok_project package, found at $TULDOK_REPOSITORY or
beside this repository (../../creative-media/Tuldok); they are skipped without it.
"""
import contextlib
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


@unittest.skipUnless(HAS_TULDOK and (LAB / 'projects' / 'rheon-model-portfolio-unified-04' / '.tuldok').is_dir(),
                     'needs Tuldok and the earlier project folder')
class LabFolder(unittest.TestCase):
    def test_committed_folder_is_up_to_date(self):
        with contextlib.redirect_stdout(io.StringIO()):
            status = converter.main(['--tuldok', str(TULDOK), '--source',
                                     str(LAB / 'projects' / 'rheon-model-portfolio-unified-04'),
                                     '--output', str(LAB / 'projects' / 'surrogate-lab'), '--check'])
        self.assertEqual(status, 0)


if __name__ == '__main__':
    unittest.main()
