"""Write the lab's Tuldok project folder (projects/surrogate-lab) from the earlier portfolio project.

A one-time conversion of projects/rheon-model-portfolio-unified-04 into the folder
format of Tuldok's docs/plans/definition-first-projects/spec-m1-m2.md. It owns
`tuldok.json`, everything under `results/` and `notes/`, and every definition that
carries `earlier_work`: the 205 earlier models as recipes, the earlier training
sets and recordings as datasets (and the first stages of the few that took two
commands, as steps), and the 31 earlier result sets as benchmarks with their
summaries. The definitions of new work in the same folder are written by hand and
are never touched.

    python scripts/build_tuldok_project.py --tuldok ../../creative-media/Tuldok \
        [--source projects/rheon-model-portfolio-unified-04] [--output projects/surrogate-lab] [--check]

`--check` writes nothing and exits 1 when the folder differs from what would be
written. Either way the script prints what it converted and what it left out.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT = {'format': 'tuldok_project_v1', 'id': '0dcc3af1e6b84b89a0e70554d98e7326',
           'name': 'Rheon surrogate lab', 'code_root': '../..'}
DEFINITION = 'tuldok_definition_v1'
KINDS = {'datasets': 'dataset', 'models': 'model', 'recipes': 'recipe', 'benchmarks': 'benchmark',
         'steps': 'step', 'environments': 'environment', 'charts': 'chart'}

NAME = re.compile(r'[a-z0-9][a-z0-9._-]{0,119}')
KEY = re.compile(r'[a-z][a-z0-9_]{0,31}')
METRIC_ID = re.compile(r'[a-z][a-z0-9_]{0,63}')

# Earlier recordings that are now new dataset definitions (surrogate/datasets/definitions).
RECORDINGS_NOW_DATASETS = {'sinusoidal-recordings': 'sinusoidal-v1', 'diversity-recordings': 'diversity-v1'}

ENVIRONMENT_OF_RUNTIME = {'managed_torch_python_v1': 'environments/cuda-torch',
                          'captured_cpu_python_v1': 'environments/lab-cpu'}
OUTPUT_ROLES = {'checkpoint': 'weights', 'predictions': 'data', 'training_splits': 'data',
                'training_diagnostics': 'log', 'training_result': 'report', 'result_summary': 'report'}
UNKNOWN_TOKEN_NOTE = {
    '<input evidence>': 'a file the earlier Tuldok run system wrote for each run listing its inputs',
    '<model evidence context>': 'a file the earlier Tuldok run system wrote with the model evidence context',
}

# How each earlier training set was prepared, as exact command lines, written by hand from the
# `command` text of notes/recipes/training-sets.json. Only sets whose text names one exact command
# line per stage are here; every other set is converted with rebuild "no" and its text kept in the
# description. A path the text left open (<settings file>, <folder>) is filled from the set's only
# settings file and with {output}. A set with two stages gets a step for the first; the second
# reads it as {inputs.first}.
PREPARATION = {
    'pressure-methods-v2-shape-view': [
        ['python', 'experiments/pressure-methods-v2/pack_shapes.py', '--output', '{output}']],
    'pressure-methods-v1-domain-view': [
        ['python', 'experiments/pressure-methods-v1/pack_domains.py']],
    'pressure-methods-v5-spectrum-view': [
        ['python', 'experiments/pressure-methods-v5/pack_spectrum.py']],
    'temporal-request-training-view-01': [
        ['python', 'experiments/full-ai-next-priorities-v1/pack_pressure_request_view.py', '--output', '{output}']],
    'spectral-training-view-01': [
        ['python', 'experiments/full-ai-tight-targets-v1/prepare_spectral_training_view.py',
         '--protocol', 'docs/plans/full-ai-tight-targets-v1/spectral-training-view-protocol-01.json',
         '--output', '{output}']],
    'recurrent-sampler-training-view-01': [
        ['python', 'experiments/full-ai-tight-targets-v1/prepare_sampler_training.py',
         '--protocol', 'docs/plans/full-ai-tight-targets-v1/recurrent-sampler-training-view-protocol-01.json',
         '--output', '{output}']],
    'shared-spectral-view-01': [
        ['python', 'experiments/full-ai-data-diversity-v1/prepare_shared_spectral_view_v1.py',
         '--protocol', 'docs/plans/full-ai-data-diversity-v1/shared-spectral-view-protocol-01.json',
         '--output', '{output}']],
    'internal-residual-view-01': [
        ['python', 'experiments/full-ai-data-diversity-v1/prepare_internal_residual_view_v1.py',
         '--protocol', 'docs/plans/full-ai-data-diversity-v1/internal-residual-view-protocol-01.json',
         '--output', '{output}']],
    'internal-residual-compact-sources': [
        ['python', 'experiments/full-ai-data-diversity-v1/pack_compact_internal_source_v1.py',
         '--protocol', 'docs/plans/full-ai-data-diversity-v1/internal-residual-compact-source-protocol-01.json',
         '--output', '{output}']],
    'macrostep-loss-view-01': [
        ['python', 'experiments/full-ai-data-diversity-v1/prepare_macrostep_loss_view_v1.py',
         '--protocol', 'docs/plans/full-ai-data-diversity-v1/macrostep-loss-view-protocol-01.json',
         '--output', '{output}']],
    'temporal-training-view-02': [
        ['python', 'experiments/full-ai-next-priorities-v1/pack_temporal.py'],
        ['python', 'experiments/full-ai-next-priorities-v1/finish_temporal_view.py']],
    'pressure-methods-v3-derived-view': [
        ['python', 'experiments/pressure-methods-v3/pack.py', '--output', '{output}'],
        ['python', 'experiments/pressure-methods-v3/pack_derived.py', '--source', '{inputs.first}',
         '--output', '{output}']],
    'current-policy-bank-02': [
        ['python', 'experiments/pressure-current-policy-v1/prepare.py', '--root', '.',
         '--config', 'experiments/pressure-current-policy-v1/configs/preparation.json', '--output', '{output}'],
        ['python', 'experiments/pressure-current-policy-v1/freeze_bank.py', '--root', '.',
         '--preparation', '{inputs.first}', '--output', '{output}']],
}
NOT_ONE_COMMAND = {
    'pressure-methods-v1-spacing-and-teacher-views': 'one entry covers four views, each made by its own command',
    'pressure-methods-v4-expansion-views': 'one entry covers two views, each made by its own command',
    'direction-banks': 'one entry covers banks made from two different settings files',
    'formula-teacher-banks': 'one entry covers four teacher banks, each made by its own script',
    'modal-views': 'one entry covers three views, each made by its own script',
    'head-calibration-and-architecture-starts': 'it took several scripts, some run once per seed generation',
    'long-training-archive': 'it took one packing command and then several project scripts',
    'water-project-inputs': 'each study ran the assembler with its own configuration',
    'sealed-water-project-inputs': 'the assemble call is only described ("as in the README")',
    'typed-pressure-probe': 'it was a direct Python call, not a command line',
    'dependency-training-inputs': 'it was a sequence of Python calls, not a command line',
}


# ---------------------------------------------------------------- small helpers

def dump(document):
    return (json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def subject_name(identifier):
    """A result-set subject id as a definition name: lower case, other characters turned into '-'."""
    name = re.sub(r'[^a-z0-9._-]', '-', identifier.lower())
    if not NAME.fullmatch(name):
        raise ValueError(f'Subject id {identifier!r} does not make a valid name ({name!r}).')
    return name


def key_of(text):
    """An input or output key made from a name or a file path."""
    key = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    if not key[:1].isalpha():
        key = 'x_' + key
    return key[:32].rstrip('_')


def escape(word):
    return word.replace('{', '{{').replace('}', '}}')


def leading_id(text):
    """'sinusoidal-recordings (train split)' -> 'sinusoidal-recordings'."""
    return text.split(' ', 1)[0]


def definition(kind, title, **fields):
    document = {'format': DEFINITION, 'kind': kind, 'title': title[:200]}
    document.update((key, value) for key, value in fields.items() if value not in (None, {}, []))
    return document


def earlier(rebuild, reason=None):
    return {'rebuild': rebuild, **({'reason': reason[:1000]} if reason else {})}


def git_blob_id(data):
    """The id Git gives a file with these bytes (what `git hash-object` prints)."""
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def study_of(path):
    """'runs/<study>/...' -> '<study>'; None for other paths."""
    parts = path.split('/')
    return parts[1] if parts[0] == 'runs' and len(parts) > 2 else None


class CodePaths:
    """Maps the paths earlier runs executed (staged copies under runs/, which Git ignores) to tracked files.

    A path is replaced by a tracked file with identical content anywhere in the repository except the
    old project folder. When several tracked files match, the one under experiments/<the run's study>/
    is preferred, then any under experiments/, then the first in path order. Files the old Tuldok's
    run wiring needed (its runtime bootstrap and manifest, contribution and extension descriptors, and
    its project document) are not lab code and are left out.
    """

    def __init__(self, root, exclude):
        self.root = Path(root)
        listing = subprocess.run(['git', 'ls-files', '-s', '-z'], cwd=self.root, check=True,
                                 capture_output=True).stdout.decode('utf-8')
        self.tracked, self.by_blob = set(), collections.defaultdict(list)
        for record in filter(None, listing.split('\0')):
            meta, path = record.split('\t', 1)
            if exclude and (path + '/').startswith(exclude.rstrip('/') + '/'):
                continue
            self.tracked.add(path)
            self.by_blob[meta.split()[1]].append(path)
        self.counts = collections.Counter()

    def generated(self, path):
        name = Path(path).name
        if 'runtime-bootstrap' in Path(path).parts[:-1] or name == 'runtime-manifest.json':
            return True
        # The runs' custody validator is a copy of the old Tuldok's recipe_process.py, put in each staged
        # project by the assemble scripts (experiments/pressure-throughput-feedback-v1/assemble.py).
        if name in ('custody-validator.py', 'custody_validator.py'):
            return True
        file = self.root / path
        if name.endswith('.json') and file.is_file():
            try:
                document = json.loads(file.read_bytes())
            except ValueError:
                return False
            if isinstance(document, dict):
                return (str(document.get('format', '')).startswith('tuldok_')
                        or (name == 'project.json' and {'recipes', 'requirements'} <= set(document)))
        return False

    def resolve(self, path):
        """('tracked' | 'rewritten' | 'generated' | 'missing', the path to write or None)."""
        if path in self.tracked:
            result = 'tracked', path
        elif self.generated(path):
            result = 'generated', None
        elif not (self.root / path).is_file():
            result = 'missing', path
        else:
            twins = self.by_blob.get(git_blob_id((self.root / path).read_bytes()), [])
            study = study_of(path)
            twins = sorted(twins, key=lambda twin: (not (study and twin.startswith(f'experiments/{study}/')),
                                                   not twin.startswith('experiments/'), twin))
            result = ('rewritten', twins[0]) if twins else ('missing', path)
        self.counts[result[0]] += 1
        return result


# ---------------------------------------------------------------- training sets and recordings

class TrainingSets:
    """notes/recipes/training-sets.json as dataset (and step) definitions."""

    def __init__(self, document, code_paths):
        self.code_paths = code_paths
        self.sets = {entry['id']: entry for entry in document['training_sets']}
        self.recordings = {entry['id']: entry for entry in document['recordings']}
        self.not_training_data = {entry['id'] for entry in document['not_training_data']}

    def reference(self, built_from):
        """The dataset an entry of `built_from` names."""
        identifier = leading_id(built_from)
        if identifier in self.sets:
            return f'datasets/{identifier}'
        if identifier in self.recordings:
            return f'datasets/{RECORDINGS_NOW_DATASETS.get(identifier, identifier)}'
        raise ValueError(f'built_from entry {built_from!r} names nothing in training-sets.json.')

    def definitions(self):
        """{relative path: document} for every training set, recording group and two-stage first step."""
        files = {}
        for identifier, entry in sorted(self.recordings.items()):
            if identifier in RECORDINGS_NOW_DATASETS:
                continue
            lines = [entry['what']]
            for field, label in (('originally_captured_by', 'Captured by'), ('scenes_defined_by', 'Scenes defined by'),
                                 ('builds_prepared_by', 'Solver builds prepared by'), ('indexed_by', 'Indexed by'),
                                 ('settings', 'Settings')):
                if entry.get(field):
                    lines.append(f"{label}: {'; '.join(entry[field])}.")
            if entry.get('probe'):
                lines.append(f"Probe: {entry['probe']}.")
            if entry.get('notes'):
                lines.append(entry['notes'])
            files[f'datasets/{identifier}.json'] = definition(
                'dataset', identifier, description=' '.join(lines)[:4000],
                earlier_work=earlier('no', 'Earlier simulated recordings whose capture commands were not recorded. '
                                           + (entry.get('notes') or '')))
        for identifier, entry in sorted(self.sets.items()):
            files.update(self._training_set(identifier, entry))
        return files

    def _training_set(self, identifier, entry):
        recorded = entry['command_recorded'] is True
        lines = [entry['what'],
                 f"Prepared by: {'; '.join(entry['prepared_by'])}.",
                 f"Built from: {'; '.join(entry['built_from']) or 'nothing earlier'}."]
        if entry.get('settings'):
            lines.append(f"Settings: {'; '.join(entry['settings'])}.")
        if entry.get('needs_models'):
            lines.append(f"Computed with the weights of: {'; '.join(entry['needs_models'])}.")
        lines.append('The earlier account gives the command as '
                     + ('recorded at the time' if recorded else
                        'reconstructed from the scripts (not recorded at the time)'
                        if entry['command_recorded'] is False else f"recorded {entry['command_recorded']}")
                     + f": {entry['command']!r}." if entry['command'] else 'The earlier account has no command.')
        if entry.get('notes'):
            lines.append(entry['notes'])
        if entry.get('folders'):
            lines.append(f"Its files sat in: {'; '.join(entry['folders'])}.")
        stages = PREPARATION.get(identifier)

        if stages is None:
            if entry['command'] is None:
                reason = 'No command was recorded or reconstructed for this training set.'
            else:
                reason = (f'There is no single command line for it: {NOT_ONE_COMMAND[identifier]}. '
                          'The account is in the description.')
            return {f'datasets/{identifier}.json': definition(
                'dataset', identifier, description=' '.join(lines)[:4000], earlier_work=earlier('no', reason))}

        # The label is about this set alone: its own command, its own code, and whether it reads model weights.
        named = sorted({stage[1] for stage in stages} | set(entry.get('settings') or []))
        missing = [path for path in named if self.code_paths.resolve(path)[0] not in ('tracked', 'rewritten')]
        if missing:
            rebuild, reason = 'no', f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in Git."
        else:
            rebuild, reason = ('after-inputs' if entry.get('needs_models') else 'yes'), None
        if any(word == 'python' for stage in stages for word in stage[:1]):
            device_note = ('environments/cuda-torch because the account says it needs CUDA'
                           if 'needs CUDA' in entry['command'] else 'environments/lab-cpu')
            lines.append(f'The account does not say which Python environment ran it; the converter chose {device_note}.')
        lines.append("The run command is the converter's reading of that text as exact command lines.")
        if not any('{output}' in word for word in stages[-1]):
            lines.append('The script writes to fixed paths under runs/, not to the output directory Tuldok gives it.')
        environment = 'environments/cuda-torch' if 'needs CUDA' in entry['command'] else 'environments/lab-cpu'
        inputs = {}
        for source in entry['built_from']:
            reference = self.reference(source)
            key = key_of(reference.split('/', 1)[1])
            if inputs.get(key, reference) != reference:
                raise ValueError(f'{identifier}: two inputs would share the key {key!r}.')
            inputs[key] = reference
        files = {}
        if len(stages) == 2:
            step = f'{identifier}-first'
            files[f'steps/{step}.json'] = definition(
                'step', f'{identifier}, first stage',
                description=f'The first of the two commands that prepared the earlier training set {identifier}; '
                            f'see datasets/{identifier}.',
                earlier_work=earlier(rebuild, reason),
                run={'command': stages[0], 'environment': environment, 'sources': [stages[0][1]]})
            inputs['first'] = f'steps/{step}'
        files[f'datasets/{identifier}.json'] = definition(
            'dataset', identifier, description=' '.join(lines)[:4000], earlier_work=earlier(rebuild, reason),
            run={'command': stages[-1], 'environment': environment, 'inputs': dict(sorted(inputs.items())),
                 'sources': named})
        return files


# ---------------------------------------------------------------- earlier models as recipes

class Recipes:
    """notes/recipes/<stem>.json (one per model card) as recipe definitions."""

    def __init__(self, folder, code_paths):
        self.code_paths = code_paths
        self.training_scripts = {}      # stem -> (state, recorded path), for the report
        self.missing_sources = {}       # stem -> recorded source paths with no tracked copy
        self.recorded = {path.stem: load(path) for path in sorted(folder.glob('*.json'))
                         if path.name != 'training-sets.json'}
        by_model = collections.defaultdict(list)
        for stem, recipe in self.recorded.items():
            by_model[recipe['model']].append(stem)
        self.stem_of_model = {model: stems[0] for model, stems in by_model.items() if len(stems) == 1}
        self.stem_of_fit = {recipe['fit_id']: stem for stem, recipe in self.recorded.items()}
        self.unknown_tokens = collections.Counter()

    def definitions(self):
        return {f'recipes/{stem}.json': self.convert(stem, recipe) for stem, recipe in self.recorded.items()}

    def producer(self, recorded_input):
        """recipes/<stem> of the earlier model whose output this input is, else None."""
        models = recorded_input.get('from_model')
        # A list means several models wrote these same bytes; any of them reproduces the file.
        for model in sorted([models] if isinstance(models, str) else models or []):
            if model in self.stem_of_model:
                return f'recipes/{self.stem_of_model[model]}'
        return None

    def convert(self, stem, recipe):
        labels = {name: recipe[name] for name in ('fit_id', 'tuldok_run', 'trained_at', 'device') if recipe.get(name)}
        for recorded_input in recipe['inputs']:
            if recorded_input.get('path'):
                labels[f"input:{recorded_input['name']}"] = recorded_input['path'][:400]
        if len(labels) > 32:
            raise ValueError(f'{stem}: more than 32 labels.')
        card = f'notes/fits/{stem}.md'
        if recipe['card'] != card:
            raise ValueError(f"{stem}: its card is {recipe['card']}, not {card}.")
        if not recipe.get('command'):
            return definition('recipe', recipe['model'], card=card, labels=labels,
                              description=f"Earlier model. {recipe.get('note', '')} Its run folder was "
                                          f"{recipe.get('run_folder')}.",
                              earlier_work=earlier('no', recipe.get('note') or 'No command was recorded.'))

        notes, inputs, paths = [], {}, {}
        weights, lost, supplied = False, [], []
        for recorded_input in recipe['inputs']:
            name, kind = recorded_input['name'], recorded_input.get('kind')
            if not KEY.fullmatch(name):
                raise ValueError(f'{stem}: input name {name!r} is not a valid key.')
            if kind == 'training data':
                inputs[name] = f"datasets/{recorded_input['training_set']}"
            elif kind == 'earlier model weights':
                weights = True
                producer = self.producer(recorded_input)
                if producer:
                    inputs[name] = producer
                else:
                    paths[name] = recorded_input['path']
                    lost.append(recorded_input['path'])
                    notes.append(f"Input {name} ({recorded_input['path']}) holds earlier weights that the account "
                                 'does not tie to a model.')
            else:
                paths[name] = recorded_input['path']
                supplied.append(f'the environment listing given as input {name}')

        # Every script and source path, mapped to the tracked file with the same content.
        entry_state, entrypoint = self.code_paths.resolve(recipe['entrypoint'])
        self.training_scripts[stem] = (entry_state, recipe['entrypoint'])
        written, missing, dropped = {}, [], []
        for source in recipe['sources']:
            if source == recipe.get('settings_file'):
                self.code_paths.counts['carried as settings'] += 1
                continue
            state, path = self.code_paths.resolve(source)
            if state == 'generated':
                dropped.append(source)
            else:
                written[source] = path
                if state == 'missing':
                    missing.append(source)

        settings_name = Path(recipe['settings_file']).name if recipe.get('settings_file') else None
        command = ['python', entrypoint]
        for token in re.findall(r'<[^>]*>|\S+', recipe['command']):
            match = re.fullmatch(r'<(input|source): (.+)>', token)
            if token == '<output dir>':
                command.append('{output}')
            elif match and match[1] == 'input':
                name = match[2]
                if name not in inputs and name not in paths:
                    raise ValueError(f'{stem}: the command reads input {name!r}, which the recipe does not list.')
                command.append(f'{{inputs.{name}}}' if name in inputs else escape(paths[name]))
            elif match and match[2] == settings_name:
                command.append('{settings}')
            elif match:
                found = [source for source in recipe['sources'] if Path(source).name == match[2]]
                if len(found) != 1:
                    raise ValueError(f'{stem}: source {match[2]!r} matches {len(found)} recorded sources.')
                if found[0] in written:
                    command.append(escape(written[found[0]]))
                else:
                    command.append(escape(token))
                    supplied.append(f'{token} (written by the old Tuldok for the run)')
            elif token.startswith('<'):
                self.unknown_tokens[token] += 1
                command.append(escape(token))
                supplied.append(f'{token} ({UNKNOWN_TOKEN_NOTE[token]})')
            else:
                command.append(escape(token))
        if len(command) > 64:
            raise ValueError(f'{stem}: the command has more than 64 words.')

        self.missing_sources[stem] = missing
        problems = []
        if entry_state == 'missing':
            problems.append(f"its training script {recipe['entrypoint']} has no tracked copy in Git")
        if missing:
            problems.append(f"{len(missing)} of its source files {'has' if len(missing) == 1 else 'have'} no tracked "
                            f'copy in Git (for example {missing[0]})')
        if lost:
            problems.append(f'it starts from weights at {lost[0]} whose producing model is not identified')
        if problems:
            label = earlier('no', 'Cannot be rebuilt: ' + '; '.join(problems) + '.')
        elif supplied:
            label = earlier('needs-work', 'The recorded command takes arguments the old Tuldok supplied: '
                            + '; '.join(dict.fromkeys(supplied)) + '. They have to be replaced before it can run.')
        else:
            label = earlier('after-inputs' if weights else 'yes')
        if dropped:
            notes.append(f'{len(dropped)} source files the old Tuldok generated for the run were left out.')

        run = {'command': command, 'environment': ENVIRONMENT_OF_RUNTIME[recipe['runtime']],
               'inputs': inputs, 'settings': recipe['settings'], 'outputs': self.outputs(recipe['outputs']),
               'resources': self.resources(recipe), 'sources': list(written.values())}
        run = {key: value for key, value in run.items() if value not in (None, {}, [])}
        description = (f"Earlier model, trained {recipe['trained_at'][:10]} by Tuldok run {recipe['tuldok_run']} "
                       f"({recipe['run_name']}). Converted from its extracted training recipe. The run executed "
                       'staged copies under runs/; each script and source is given as the tracked file with the '
                       'same content where one exists.')
        if notes:
            description += ' ' + ' '.join(dict.fromkeys(notes))
        return definition('recipe', recipe['model'], description=description[:4000], card=card, labels=labels,
                          earlier_work=label, run=run)

    @staticmethod
    def outputs(recorded):
        outputs = {}
        for output in recorded:
            key = key_of(output['path'])
            if key in outputs:
                raise ValueError(f"Two outputs share the key {key!r}.")
            outputs[key] = {'path': output['path'], 'role': OUTPUT_ROLES.get(output['role'], 'file')}
        return outputs

    @staticmethod
    def resources(recipe):
        recorded = recipe.get('resources') or {}
        resources = {}
        if recipe.get('device') == 'cuda':
            resources['gpu'] = True
        if recorded.get('cpu_cores'):
            resources['cpu_threads'] = int(recorded['cpu_cores'])
        if recorded.get('memory_bytes'):
            resources['memory_gb'] = round(recorded['memory_bytes'] / 1e9, 3)
        if recorded.get('wall_seconds'):
            resources['wall_hours'] = round(recorded['wall_seconds'] / 3600, 4)
        if recorded.get('max_output_bytes'):
            resources['disk_gb'] = round(recorded['max_output_bytes'] / 1e9, 4)
        return resources


# ---------------------------------------------------------------- earlier result sets as benchmarks

def result_set_group(title):
    """The benchmark a result set belongs to: its title without the subject range, subject count and part."""
    title = re.sub(r'^C\d+–C\d+ · ', '', title)
    title = re.sub(r' \(\d+ subjects\)$', '', title)
    return re.sub(r' · part \d+$', '', title)


def dimension_text(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    text = repr(value) if isinstance(value, float) else str(value)
    if re.search(r'[;=\r\n]', text):
        raise ValueError(f'Dimension value {text!r} cannot be written in a group.')
    return text


class SubjectMatcher:
    """Finds the earlier model behind a result-set subject, only where the project's own records say so."""

    def __init__(self, notes, stem_of_fit):
        self.stem_of_fit = stem_of_fit
        self.aliases = collections.defaultdict(list)
        for alias in load(notes / 'portfolio-index.json')['aliases']:
            if alias['fit_item_id']:
                self.aliases[alias['label']].append(alias)

    def match(self, subject):
        """(recipe reference or None, how it was matched or why not)."""
        dimensions = subject['dimensions']
        if not isinstance(dimensions, dict):
            dimensions = {entry['id']: entry['value'] for entry in dimensions}
        fit = dimensions.get('fit_item_id')
        route = 'fit_item_id'
        if not fit and str(dimensions.get('fit_or_execution_id') or '').startswith('fit-'):
            fit, route = dimensions['fit_or_execution_id'], 'fit_or_execution_id'
        if not fit and dimensions.get('weights_sha256'):
            fits = {alias['fit_item_id'] for alias in self.aliases.get(subject['id'], [])
                    if alias['declared_weights_sha256'] == dimensions['weights_sha256']}
            if len(fits) == 1:
                fit, route = fits.pop(), 'checkpoint label and weights checksum in notes/portfolio-index.json'
        if not fit:
            if dimensions.get('weights_sha256'):
                return None, 'its weights checksum is not linked to a training record in the notes'
            if dimensions.get('origin_kind') == 'assembly_execution':
                return None, 'a solver assembly, not a trained model'
            if dimensions.get('parent_fit_item_id'):
                return None, f"an execution variant built on {dimensions['parent_fit_item_id']}, not a model of its own"
            return None, 'it carries no model identity'
        if fit not in self.stem_of_fit:
            return None, f'{fit} has no recipe'
        return f'recipes/{self.stem_of_fit[fit]}', route


class Benchmarks:
    """The project's measurement_source result sets, grouped into benchmark definitions and summarised."""

    def __init__(self, source, matcher):
        self.source = source
        self.matcher = matcher
        blobs = {}
        for directory, _, names in os.walk(source / 'artifacts' / 'sha256'):
            for name in names:
                blobs[name.split('.', 1)[0]] = Path(directory) / name
        inventory = load(source / '.tuldok' / 'project.json')['inventory']
        self.result_sets = []
        for entry in sorted((e for e in inventory if e.get('family') == 'measurement_source'), key=lambda e: e['path']):
            record = load(source / '.tuldok' / entry['path'])
            found = [path for name, path in blobs.items() if name.startswith(record['blob']['sha256'])]
            if len(found) != 1:
                raise FileNotFoundError(f"Result set {record['title']!r}: {len(found)} stored files match its checksum.")
            self.result_sets.append({'title': record['title'], 'imported_at': record['imported_at'], 'file': found[0]})
        self.matches = []   # (benchmark, subject id, recipe reference or None, route or reason)

    def build(self, summarise, write_results, summary_row):
        groups = collections.defaultdict(list)
        for result_set in self.result_sets:
            groups[result_set_group(result_set['title'])].append(result_set)
        files = {}
        for title, members in sorted(groups.items()):
            name = slug(title)
            document, rows = self._benchmark(name, title, members, summarise, summary_row)
            files[f'benchmarks/{name}.json'] = document
            files[f'results/{name}.csv'] = write_results(rows).encode('utf-8')
        return files

    def _benchmark(self, name, title, members, summarise, summary_row):
        metrics, subjects, cases, scopes = {}, {}, set(), set()
        values = collections.defaultdict(list)          # (subject, metric, group items) -> measured values
        unavailable = collections.Counter()
        produced = {}
        dimension_values = collections.defaultdict(set)
        for member in members:
            document = load(member['file'])
            scopes.add((document['scope']['id'], document['scope']['label'], document['scope'].get('population')))
            for metric in document['metrics']:
                entry = {'id': metric['id'].replace('-', '_'), 'title': metric['label'], 'units': metric['units'],
                         'better': 'lower'}
                if not METRIC_ID.fullmatch(entry['id']) or not re.search(r'l2|rms|max|ms', entry['id']):
                    raise ValueError(f"Metric {metric['id']!r} of {title!r} needs a rule for its name or direction.")
                if metrics.setdefault(metric['id'], entry) != entry:
                    raise ValueError(f"Metric {metric['id']!r} differs between the result sets of {title!r}.")
            for subject in document['subjects']:
                if subject['id'] in subjects:
                    raise ValueError(f"Subject {subject['id']!r} appears in two result sets of {title!r}.")
                subjects[subject['id']] = subject
                produced[subject['id']] = member['imported_at'][:10]
            cases.update(case['id'] for case in document['cases'])
            for observation in document['observations']:
                dimensions = observation.get('dimensions') or {}
                if isinstance(dimensions, list):
                    dimensions = {entry['id']: entry['value'] for entry in dimensions}
                group = tuple(sorted((key, dimension_text(value)) for key, value in dimensions.items()))
                for key, text in group:
                    dimension_values[key].add(text)
                key = (observation['subject_id'], observation['metric_id'], group)
                if observation['state'] == 'measured' and observation['value'] is not None:
                    values[key].append(observation['value'])
                else:
                    unavailable[key] += 1
            del document
        if len(scopes) != 1:
            raise ValueError(f'The result sets grouped as {title!r} have different scopes: {sorted(scopes)}.')
        (scope_id, scope_label, population), = scopes
        # Group by the per-value dimensions that actually split the values.
        group_by = sorted(key for key, texts in dimension_values.items() if len(texts) > 1)
        for key in group_by:
            if not METRIC_ID.fullmatch(key):
                raise ValueError(f'Dimension {key!r} of {title!r} cannot be a group_by key.')

        # One row per subject and metric over all values, and, when there is a grouping, one per group as well.
        measured_in, missing_in = collections.defaultdict(list), collections.Counter()
        for subject, metric, group in sorted(set(values) | set(unavailable)):
            chosen = tuple(pair for pair in group if pair[0] in group_by)
            for text in sorted({'', ';'.join(f'{key}={value}' for key, value in chosen)}):
                measured_in[subject, metric, text].extend(values.get((subject, metric, group), []))
                missing_in[subject, metric, text] += unavailable.get((subject, metric, group), 0)
        summary_rows = []
        for (subject, metric, group) in sorted(set(measured_in) | set(missing_in)):
            summary = summarise(measured_in[subject, metric, group], missing_in[subject, metric, group])
            summary_rows.append(summary_row(
                subject=subject_name(subject), metric=metric.replace('-', '_'), group=group, commit='',
                produced=produced[subject], **summary))

        subject_entries, names = [], set()
        for identifier in sorted(subjects):
            subject = subjects[identifier]
            entry = {'name': subject_name(identifier)}
            if entry['name'] in names:
                raise ValueError(f'Two subjects of {title!r} become the name {entry["name"]!r}.')
            names.add(entry['name'])
            label = subject.get('label') or identifier
            entry['title'] = (identifier if label == identifier else f'{identifier} · {label}')[:200]
            recipe, route = self.matcher.match(subject)
            self.matches.append((name, identifier, recipe, route))
            if recipe:
                entry['recipe'] = recipe
            subject_entries.append(entry)
        description = (f"Earlier benchmark '{scope_label}' (scope {scope_id}, population {population}), measured by "
                       f'the earlier study harness on {len(cases)} cases. Converted from {len(members)} result '
                       f"set{'s' if len(members) > 1 else ''} titled "
                       + '; '.join(f"{t!r}" + (f' ({n} sets)' if n > 1 else '')
                                   for t, n in collections.Counter(m['title'] for m in members).items())
                       + f'. Summaries are in results/{name}.csv; the per-case values stay in the earlier '
                       "project's store.")
        document = definition(
            'benchmark', title, description=description[:4000],
            earlier_work=earlier('no', 'Measured by the earlier study harness, which is not a command this '
                                       'project can run.'),
            metrics=list(metrics.values()), subjects=subject_entries, cases={'count': len(cases)},
            summary={'group_by': group_by} if group_by else None)
        return document, summary_rows


# ---------------------------------------------------------------- what the notes hold that is not carried

def notes_left_out(notes):
    """Plain sentences on what notes/*.json, notes/checkpoints/ and notes/recipes/ hold that the new folder does not."""
    index = load(notes / 'portfolio-index.json')
    portfolio = load(notes / '00-models.portfolio.json')
    checkpoints = [row for path in sorted((notes / 'checkpoints').glob('*.json')) for row in load(path)['rows']]
    dispositions = collections.Counter(alias['fit_link_disposition'] for alias in index['aliases'])
    return [
        f"notes/portfolio-index.json: {len(index['aliases'])} checkpoint aliases (dashboard labels such as C01 with "
        f"declared weights checksums, candidate fits and link dispositions: "
        + ', '.join(f'{count} {name}' for name, count in dispositions.most_common()) + '); '
        f"{len(index['fits'])} fit entries with their checkpoint checksums by stage (selected, initial, retained "
        "snapshots), matched-benchmark pointers, method review status and card checksums; "
        f"{len(index['source_candidates'])} model source-file candidates with class listings; the pressure bundle "
        "summary, source pointers, definitions of alias/checkpoint/fit, and "
        f"{len(index['limitations'])} stated limitations. Only the alias-to-fit links were used here, to match "
        'subjects; the rest is not carried.',
        f"notes/00-models.portfolio.json: {len(portfolio['models'])} model entries (aliases, review state, task "
        f"group, a long method summary), {len(portfolio['groups'])} authored task groups, and "
        f"{len(portfolio['relationships'])} authored comparisons between fits with excerpts. The model "
        'summaries partly repeat the cards; the groups and comparisons are not carried.',
        f"notes/checkpoints/ ({len(checkpoints)} rows): one row per distinct checkpoint file (checksum, size, the "
        'retained path where it was verified, every path it was found at, and its context). It is the map from '
        'weights checksums to files on the training machine, which registering the earlier weights as products '
        'will need. Not carried.',
        'notes/recipes/*.json and training-sets.json: converted into recipes/ and datasets/. Not carried: run names '
        "as a field (they are in each recipe's description), the earlier runtime name, input sizes, the output "
        "roles' finer names, the recorded settings_file path, the folders' command_recorded_in pointers and "
        'not_training_data. Each recipe keeps the recorded path of every input as an input:<name> label.',
    ]


# ---------------------------------------------------------------- the folder

def owned_by_script(relative, data):
    """Whether this script wrote a file: tuldok.json, results/, notes/, or a definition with earlier_work."""
    top = relative.split('/', 1)[0]
    if relative == 'tuldok.json' or top in ('results', 'notes'):
        return True
    if top in KINDS and relative.endswith('.json'):
        try:
            return 'earlier_work' in json.loads(data)
        except ValueError:
            return False
    return False


def existing_files(output):
    files = {}
    if output.is_dir():
        for path in sorted(output.rglob('*')):
            if path.is_file():
                files[path.relative_to(output).as_posix()] = path.read_bytes()
    return files


def build(source, output, tuldok):
    """Everything the script owns, as {relative path: bytes}, plus what the report needs."""
    sys.path.insert(0, str(Path(tuldok).resolve()))
    from tuldok_project.results import SummaryRow, summarise, write_results

    notes = source / 'notes'
    code_root = (Path(output) / PROJECT['code_root']).resolve()
    old_project = Path(source).resolve().relative_to(code_root).as_posix()
    code_paths = CodePaths(code_root, exclude=old_project)
    training_sets = TrainingSets(load(notes / 'recipes' / 'training-sets.json'), code_paths)
    recipes = Recipes(notes / 'recipes', code_paths)
    benchmarks = Benchmarks(source, SubjectMatcher(notes, recipes.stem_of_fit))

    documents = {**training_sets.definitions(), **recipes.definitions()}
    files = {path: dump(document) for path, document in documents.items()}
    for path, content in benchmarks.build(summarise, write_results, SummaryRow).items():
        files[path] = content if isinstance(content, bytes) else dump(content)
    files['tuldok.json'] = dump(PROJECT)
    for path in sorted([*notes.glob('*.md'), *(notes / 'fits').glob('*.md')]):
        files[path.relative_to(source).as_posix()] = path.read_bytes()

    hand_written = {path for path, data in existing_files(output).items() if not owned_by_script(path, data)}
    clash = sorted(set(files) & hand_written)
    if clash:
        raise ValueError(f'These files are written by hand and the script would overwrite them: {clash}')
    return files, {'benchmarks': benchmarks, 'recipes': recipes, 'notes': notes, 'code_paths': code_paths}


SEVERITY = ('new work', 'yes', 'after-inputs', 'needs-work', 'no')


def effective_labels(files):
    """Counts of (kind, label) where each item takes the worst label of itself and everything upstream of its inputs."""
    documents = {path[:-len('.json')]: json.loads(data) for path, data in files.items()
                 if path.split('/', 1)[0] in KINDS and path.endswith('.json')}
    worst = {}

    def label(reference):
        if reference not in worst:
            worst[reference] = 'new work'      # guards against cycles; the checker reports those
            document = documents.get(reference, {})     # a missing definition is the checker's to report
            own = document.get('earlier_work', {}).get('rebuild', 'new work')
            upstream = [label(item if isinstance(item, str) else item['from'])
                        for item in (document.get('run', {}).get('inputs') or {}).values()]
            worst[reference] = max([own, *upstream], key=SEVERITY.index)
        return worst[reference]

    return collections.Counter((documents[r]['kind'], label(r)) for r in sorted(documents))


def report(files, output, details, size):
    existing = existing_files(output)
    hand = {path: data for path, data in existing.items() if not owned_by_script(path, data)}
    everything = {**hand, **files}
    kinds, labels = collections.Counter(), collections.Counter()
    for path, data in sorted(everything.items()):
        top = path.split('/', 1)[0]
        if top in KINDS and path.endswith('.json'):
            document = json.loads(data)
            kinds[KINDS[top]] += 1
            rebuild = document.get('earlier_work', {}).get('rebuild', 'new work')
            labels[(KINDS[top], rebuild)] += 1
    lines = ['Definitions by kind:']
    lines += [f'  {kind:12s} {kinds[kind]:4d}' for kind in KINDS.values() if kinds[kind]]
    lines.append('Rebuild labels:')
    lines += [f'  {kind:12s} {label:13s} {count:4d}' for (kind, label), count in sorted(labels.items())]
    lines.append('Effective labels, carrying no and needs-work up through inputs:')
    effective = effective_labels(everything)
    lines += [f'  {kind:12s} {label:13s} {count:4d}' for (kind, label), count in sorted(effective.items())]
    counts = details['code_paths'].counts
    lines.append(f"Code paths of earlier items: {counts['rewritten']} rewritten to a tracked file with the same "
                 f"content, {counts['tracked']} already tracked, {counts['missing']} with no tracked file, "
                 f"{counts['generated']} generated by the old Tuldok and left out, {counts['carried as settings']} "
                 'settings files carried as the recipe\'s settings.')
    scripts = details['recipes'].training_scripts
    absent = collections.defaultdict(list)
    for stem, (state, path) in sorted(scripts.items()):
        if state == 'missing':
            absent[study_of(path) or 'not under runs/'].append(stem)
    lines.append(f'Earlier recipes whose training script is not in Git: {sum(map(len, absent.values()))} '
                 f'of {len(scripts)} with a command.')
    for study, stems in sorted(absent.items()):
        lines.append(f'  {study} ({len(stems)}):')
        lines += [f'    {stem}' for stem in stems]
    missing = {stem: paths for stem, paths in details['recipes'].missing_sources.items() if paths}
    code_missing = [stem for stem, paths in missing.items() if any(p.endswith('.py') for p in paths)]
    names = collections.Counter(Path(p).name for paths in missing.values() for p in set(paths))
    lines.append(f'Earlier recipes with sources that have no tracked copy: {len(missing)}; {len(code_missing)} of them '
                 f'miss Python code, {len(missing) - len(code_missing)} only data or settings files. Most common: '
                 + ', '.join(f'{name} ({count})' for name, count in names.most_common(12)) + '.')
    matches = details['benchmarks'].matches
    matched = [m for m in matches if m[2]]
    routes = collections.Counter(m[3] for m in matched)
    lines.append(f'Result-set subjects: {len(matches)}; matched to a model {len(matched)}, '
                 f'not matched {len(matches) - len(matched)}.')
    lines += [f'  matched by {route}: {count}' for route, count in sorted(routes.items())]
    lines.append('  Not matched:')
    lines += [f'    {benchmark}: {subject} ({reason})' for benchmark, subject, recipe, reason in matches if not recipe]
    tokens = details['recipes'].unknown_tokens
    if tokens:
        lines.append('Command words with no equivalent now, kept as literal text: '
                     + ', '.join(f'{token} in {count} recipes' for token, count in sorted(tokens.items())))
    results = sum(len(data) for path, data in files.items() if path.startswith('results/'))
    notes_size = sum(len(data) for path, data in files.items() if path.startswith('notes/'))
    lines.append(f'Folder size: {size / 1e6:.2f} MB in {len(everything)} files '
                 f'(notes {notes_size / 1e6:.2f} MB, results {results / 1e6:.2f} MB).')
    lines.append('Left in the old notes and not carried:')
    lines += [f'  - {line}' for line in notes_left_out(details['notes'])]
    return '\n'.join(lines)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--tuldok', required=True, type=Path, help='the Tuldok repository, to import tuldok_project')
    parser.add_argument('--source', type=Path, default=Path('projects/rheon-model-portfolio-unified-04'))
    parser.add_argument('--output', type=Path, default=Path('projects/surrogate-lab'))
    parser.add_argument('--check', action='store_true', help='write nothing; fail if the folder differs')
    options = parser.parse_args(arguments)

    files, details = build(options.source, options.output, options.tuldok)
    existing = existing_files(options.output)
    stale = sorted(path for path, data in existing.items() if owned_by_script(path, data) and path not in files)
    changed = sorted(path for path, data in files.items() if existing.get(path) != data)

    if options.check:
        size = sum(len(data) for path, data in existing.items())
        print(report(files, options.output, details, size))
        if changed or stale:
            print(f'\nOut of date: {len(changed)} files differ or are missing, {len(stale)} should not be there.')
            for path in (changed + stale)[:20]:
                print(f'  {path}')
            return 1
        print('\nThe folder matches what the script writes.')
        return 0

    for path in stale:
        (options.output / path).unlink()
    for path in changed:
        target = options.output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[path])
    for directory in sorted((p for p in options.output.rglob('*') if p.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    size = sum(len(data) for data in existing_files(options.output).values())
    print(report(files, options.output, details, size))
    print(f'\nWrote {len(changed)} files, removed {len(stale)}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
