"""Write one readable training recipe per model of a Tuldok project.

Tuldok keeps the exact account of every training run inside its run records:
the command, every source file and every input, each as a stored file. That
account is complete but hard to read, and the settings file of a run is only in
Git when the run's model source happens to include it. This script lifts out
what a person or a tool needs to retrain a model:

    notes/recipes/<model>.json
        model, card           which model this is
        command, entrypoint   what was run
        settings              the training settings (the run's config.json)
        inputs                the prepared files it read; `from_model` names the earlier
                              model whose weights an input is, and `training_set` names the
                              entry of notes/recipes/training-sets.json that says which
                              script prepared the file and from what
        sources               the code and settings files the run used
        outputs, device, ...

It needs the project's stored files to read the settings, so run it on the
machine that holds them, after the project gains models, and commit the result.

Usage: python scripts/extract_model_recipes.py projects/<project-directory> [--check]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RUN_ID = re.compile(r'(?:\.external-|/workspace-runs/)([0-9a-f]{32})')
# A run republishes the weights it started from under these names; they are its parent's, not its own.
STARTING_WEIGHTS = re.compile(r'^(initial-weights|checkpoint-0+)\.npz$')
_TOKENS = (('literal', '{}'), ('source', '<source: {}>'), ('input', '<input: {}>'), ('parameter', '<parameter: {}>'))


def _load(path):
    return json.loads(Path(path).read_bytes())


def _command(arguments):
    """The recipe's argument list as text, with named placeholders for files and directories."""
    words = []
    for token in arguments:
        for key, pattern in _TOKENS:
            if key in token:
                value = token[key]
                words.append(pattern.format(Path(value).name if key == 'source' else value))
                break
        else:
            words.append('<' + next(iter(token)).replace('_', ' ') + '>')
    return ' '.join(words)


def _settings_source(recipe, sources):
    """The stored name of the settings file: the source given to --config, else the only config.json."""
    arguments = recipe.get('arguments', [])
    for flag, value in zip(arguments, arguments[1:]):
        if flag.get('literal') == '--config' and 'source' in value:
            return 'source/' + value['source']
    named = [name for name in sources if name.endswith('/config.json')]
    return named[0] if len(named) == 1 else None


class TrainingSets:
    """notes/recipes/training-sets.json: the hand-traced account of how each prepared input was produced."""

    def __init__(self, path):
        self.document = _load(path) if Path(path).is_file() else {'training_sets': [], 'not_training_data': []}
        self.by_folder, self.patterns = {}, []
        for kind, key in (('training data', 'training_sets'), ('not training data', 'not_training_data')):
            for entry in self.document[key]:
                for folder in entry.get('folders', []):
                    self.by_folder[folder] = (kind, entry['id'])
                if 'folder_pattern' in entry:
                    self.patterns.append((re.compile(entry['folder_pattern']), kind, entry['id']))

    def find(self, path):
        """(kind, id) for the folder a file was read from, or (None, None)."""
        folder = str(Path(path).parent) if '/' in path else ''
        if folder in self.by_folder:
            return self.by_folder[folder]
        for pattern, kind, identity in self.patterns:
            if pattern.search(folder):
                return kind, identity
        return None, None

    def scripts(self):
        """Every repository path the document names as a script or settings file."""
        entries = self.document['training_sets'] + self.document.get('recordings', [])
        keys = ('prepared_by', 'views_prepared_by', 'settings', 'originally_captured_by', 'indexed_by',
                'builds_prepared_by')
        return sorted({item.split(' (')[0].split(',')[0] for entry in entries for key in keys
                       for item in entry.get(key, [])})


class Project:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.training_sets = TrainingSets(self.directory / 'notes' / 'recipes' / 'training-sets.json')
        manifest = _load(self.directory / '.tuldok' / 'project.json')
        self.runs = {}
        for entry in manifest['inventory']:
            if entry['family'] == 'run_history':
                record = _load(self.directory / '.tuldok' / entry['path'])
                self.runs[record['run']['id']] = record
        self.fits = _load(self.directory / 'notes' / 'portfolio-index.json')['fits']
        self.checkpoint_paths = {}
        for part in sorted((self.directory / 'notes' / 'checkpoints').glob('part-*.json')):
            for row in _load(part)['rows']:
                self.checkpoint_paths[row['sha256']] = row['verified_retained_path']
        self.training_run = {}      # training receipt -> run id
        self.weights_from = {}      # stored file -> runs that trained it (not runs that merely started from it)
        for run_id, record in self.runs.items():
            for payload in record['payloads']:
                if payload['role'] == 'training_result':
                    self.training_run.setdefault(payload['blob']['sha256'], run_id)
                elif payload['role'] == 'checkpoint' and not STARTING_WEIGHTS.match(payload['name']):
                    self.weights_from.setdefault(payload['blob']['sha256'], set()).add(run_id)
        self.run_of_fit = {fit['item_id']: self._run_of(fit) for fit in self.fits}
        self.fit_of_run = {run_id: fit for fit in self.fits
                           if (run_id := self.run_of_fit[fit['item_id']]) in self.runs}

    def _checkpoints(self, fit):
        return [c if isinstance(c, str) else c['sha256'] for c in fit.get('checkpoints') or []]

    def _run_of(self, fit):
        """The run that trained a fit: by its training receipt, else by the run folder its weights were kept in."""
        if fit['receipt_sha256'] in self.training_run:
            return self.training_run[fit['receipt_sha256']]
        for sha in self._checkpoints(fit):
            found = RUN_ID.search(self.checkpoint_paths.get(sha, ''))
            if found:
                return found.group(1)
        return None

    def stored(self, sha256):
        path = self.directory / 'artifacts' / 'sha256' / sha256
        return path if path.is_file() else None

    def _folder_recipe(self, fit):
        """A recipe read from the run's own folder, for the few runs that predate this project's records."""
        folders = [Path(self.checkpoint_paths[sha]).parent.parent for sha in self._checkpoints(fit)
                   if sha in self.checkpoint_paths]
        if not folders:
            return None, 'no run record and no known run folder'
        folder = max(set(folders), key=folders.count)
        here = self.directory.resolve().parents[1] / folder
        if not (here / 'source' / 'config.json').is_file():
            return None, f'its training run is not among the project\'s run records and {folder} is not on this machine'
        listing = lambda name: sorted(path.name for path in (here / name).iterdir()) if (here / name).is_dir() else []
        return {
            'model': fit['name'], 'fit_id': fit['item_id'], 'card': fit['card_path'],
            'tuldok_run': None, 'run_folder': str(folder),
            'note': 'Trained before runs were recorded in this project; read from the run\'s own folder, '
                    'which does not record the command line.',
            'entrypoint': 'train.py' if 'train.py' in listing('source') else None, 'command': None,
            'settings_file': 'config.json', 'settings': _load(here / 'source' / 'config.json'),
            'inputs': [{'name': name, 'path': None, 'bytes': (here / 'input' / name).stat().st_size}
                       for name in listing('input') if (here / 'input' / name).is_file()],
            'sources': listing('source'),
            'outputs': [{'path': name, 'role': None} for name in listing('output')],
        }, None

    def recipe(self, fit):
        """The recipe of one fit, or (None, reason)."""
        run_id = self.run_of_fit[fit['item_id']]
        if run_id not in self.runs:
            return self._folder_recipe(fit)
        record = self.runs[run_id]
        run, material = record['run'], record['run']['material']
        definition = material['recipe']
        files = {p['name']: p['blob'] for p in record['payloads'] if p['role'] == 'execution_source'}
        sources = sorted(name for name in files if name.startswith('source/'))
        settings_name = _settings_source(definition, sources)
        settings = None
        if settings_name:
            stored = self.stored(files[settings_name]['sha256'])
            if stored is None:
                return None, 'its settings file is not in this machine\'s store'
            settings = _load(stored)
        inputs = []
        for name, reference in material['inputs'].items():
            blob = files.get('input/' + name, {})
            entry = {'name': name, 'path': reference.get('path'), 'bytes': blob.get('bytes')}
            parents = sorted({self.fit_of_run[r]['name'] for r in self.weights_from.get(blob.get('sha256'), ())
                              if r in self.fit_of_run and r != run_id})
            kind, training_set = self.training_sets.find(reference.get('path') or '')
            if parents:
                entry['kind'] = 'earlier model weights'
                entry['from_model'] = parents[0] if len(parents) == 1 else parents
            elif training_set == 'starting-weights':
                entry['kind'] = 'earlier model weights'
            elif training_set:
                entry['kind'] = kind
                entry['training_set'] = training_set
            inputs.append(entry)
        return {
            'model': fit['name'], 'fit_id': fit['item_id'], 'card': fit['card_path'],
            'tuldok_run': run_id, 'run_name': run['graph']['name'], 'trained_at': run['created_at'],
            'device': (run.get('execution_resources') or {}).get('device') or definition.get('runtime_device'),
            'runtime': definition.get('runtime'), 'resources': definition.get('resources'),
            'entrypoint': definition.get('entrypoint'), 'command': _command(definition.get('arguments', [])),
            'parameters': material.get('parameters') or {},
            'settings_file': settings_name[len('source/'):] if settings_name else None, 'settings': settings,
            'inputs': inputs,
            'sources': [name[len('source/'):] for name in sources],
            'outputs': [{'path': output['path'], 'role': output['role']} for output in definition.get('outputs', [])],
        }, None


def render(recipe):
    return json.dumps(recipe, indent=1, ensure_ascii=False) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('project', type=Path)
    parser.add_argument('--check', action='store_true', help='fail if a recipe on disk is missing or out of date')
    arguments = parser.parse_args()
    project = Project(arguments.project)
    target = arguments.project / 'notes' / 'recipes'
    written, stale, skipped = 0, [], []
    for fit in project.fits:
        recipe, reason = project.recipe(fit)
        path = target / (Path(fit['card_path']).stem + '.json')
        if recipe is None:
            skipped.append(f"{fit['name']} ({fit['item_id']}): {reason}")
            continue
        text = render(recipe)
        if arguments.check:
            if not path.is_file() or path.read_text() != text:
                stale.append(str(path))
            continue
        target.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written += 1
    for line in skipped:
        print('no recipe: ' + line, file=sys.stderr)
    repository = arguments.project.resolve().parents[1]
    tracked = set(subprocess.run(['git', 'ls-files'], cwd=repository, check=True, capture_output=True,
                                 text=True).stdout.splitlines())
    absent = [path for path in project.training_sets.scripts() if path not in tracked]
    for path in absent:
        print(f'training-sets.json names {path}, which is not in the repository', file=sys.stderr)
    if arguments.check:
        for path in stale:
            print(f'{path} is missing or out of date; run without --check.', file=sys.stderr)
        return 1 if stale or absent else 0
    print(f'{target}: {written} recipes written, {len(skipped)} models without one')
    return 0


if __name__ == '__main__':
    sys.exit(main())
