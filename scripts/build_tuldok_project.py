"""Write the lab's Tuldok project folder (projects/surrogate-lab) from the earlier portfolio project.

The conversion is done. projects/surrogate-lab is now the lab's Tuldok project, edited
directly or through Tuldok and committed like any source, and it is no longer required
to equal this script's output. The script is kept as the record of how that folder was
made from projects/rheon-model-portfolio-unified-04 (the earlier folder, no longer in
Git), and to write the adoption manifest (`--adoption-output`) and the values files
(`--values-output`). It needs the earlier folder on disk, so it runs only on the
machine that has it. Run without `--check`, it still rewrites the files it owns in
--output, so a hand edit to one of them would be overwritten.

The conversion was into the folder format of Tuldok's
docs/plans/definition-first-projects/spec-m1-m2.md. The script owns `tuldok.json`,
everything under `results/` and `notes/`, and every definition that carries
`earlier_work`: the 205 earlier models as recipes, the earlier training sets and
recordings as datasets (and the first stages of the few that took two commands, as
steps), and the 31 earlier result sets as benchmarks with their summaries. The
definitions of new work in the same folder are written by hand and are never touched.

The map fields (Tuldok's docs/plans/definition-first-projects/spec-w1.md) are written too:
`groups` and `map.chart` in tuldok.json, and on the earlier recipes `group` and `compared_with`
(the authored groups and comparisons of the old notes/00-models.portfolio.json) and `change`,
`derived_from` and the label `short` (from scripts/earlier-model-map.json, see its `about`).
`map.chart` names a hand-written chart of the folder.

    python scripts/build_tuldok_project.py --tuldok ../../creative-media/Tuldok \
        [--source projects/rheon-model-portfolio-unified-04] [--output projects/surrogate-lab] [--check] \
        [--values-output [DIR]] [--adoption-output [FILE]] [--seed FILE]

`--check` writes nothing and exits 1 when the folder differs from what would be
written. Either way the script prints what it converted and what it left out.
`--adoption-output` also writes the manifest for Tuldok's `adopt` that makes the files
the earlier runs left in the old folder's artifacts/sha256/ the products of their
recipes, and prints what it covers and what it leaves.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

PROJECT = {'format': 'tuldok_project_v1', 'id': '0dcc3af1e6b84b89a0e70554d98e7326',
           'name': 'Rheon surrogate lab', 'code_root': '../..'}
DEFINITION = 'tuldok_definition_v1'
KINDS = {'datasets': 'dataset', 'models': 'model', 'recipes': 'recipe', 'benchmarks': 'benchmark',
         'steps': 'step', 'environments': 'environment', 'charts': 'chart'}

NAME = re.compile(r'[a-z0-9][a-z0-9._-]{0,119}')
KEY = re.compile(r'[a-z][a-z0-9_]{0,31}')
METRIC_ID = re.compile(r'[a-z][a-z0-9_]{0,63}')
GROUP_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')

# change, derived_from and the label short of each earlier model, by fit id (see its `about`).
SEED_DEFAULT = Path(__file__).resolve().parent / 'earlier-model-map.json'
# The map's track: the worst pressure error on the 128 consumed fitting requests, a hand-written chart.
MAP_CHART = 'charts/pressure-fits-worst-error-progression'

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


def card_paragraph(text):
    """(label, plain text) of a card's '**Purpose.**' paragraph, else of its '**Method.**' one, else (None, '').

    The label is dropped, the Markdown emphasis marks (**, __, `, and * or _ around words) are removed and
    the whitespace is collapsed to one line."""
    paragraphs = re.split(r'\n\s*\n', text)
    for label in ('Purpose', 'Method'):
        for paragraph in paragraphs:
            if paragraph.startswith(f'**{label}.**'):
                plain = re.sub(r'\*\*|__|`|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)', '', paragraph[len(label) + 5:])
                return label, ' '.join(plain.split())
    return None, ''


def sentences_within(text, room):
    """text, or its longest start that ends a sentence and fits in room characters ('' if none does)."""
    if len(text) <= room:
        return text
    ends = [match.end() for match in re.finditer(r'[.!?](?=\s|$)', text) if match.end() <= room]
    return text[:ends[-1]] if ends else ''


def earlier(rebuild, reason=None):
    return {'rebuild': rebuild, **({'reason': reason[:1000]} if reason else {})}


def git_blob_id(data):
    """The id Git gives a file with these bytes (what `git hash-object` prints)."""
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def stored_files(source):
    """{checksum: path} of every file in the old folder's artifacts/sha256/, which names each file by its checksum."""
    blobs = {}
    for directory, _, names in os.walk(Path(source) / 'artifacts' / 'sha256'):
        for name in names:
            blobs[name.split('.', 1)[0]] = Path(directory) / name
    return blobs


def study_of(path):
    """'runs/<study>/...' -> '<study>'; None for other paths."""
    parts = path.split('/')
    return parts[1] if parts[0] == 'runs' and len(parts) > 2 else None


EARLIER_RUN_CODE = 'earlier-run-code'

# What each untracked JSON file the earlier runs named is, decided by opening it (see the slice L2 report).
# 'settings' are values a run was configured with or read as configuration: they are copied into
# earlier-run-code/. 'record' files are records about other files or about a run (checksums,
# witnesses, receipts, qualification and provenance notes, offline probe results): they are not
# copied, are left out of `sources`, and are named in the reason when a command reads one.
JSON_FILES = {
    'architecture-variant.json': 'settings',       # variant, updates, checkpoints, stages, modes, head width
    'config.json': 'settings',                     # a run's experiment configuration (other than the recipe's own)
    'configuration.json': 'settings',              # architecture and scales of an exported pressure core
    'distillation.json': 'settings',               # distillation objective (width fractions, weight); embeds the teacher's tensor checksums
    'fine-tuning.json': 'settings',                # fine-tuning objective (passes, teacher weight); embeds the starting tensors' checksums
    'fit-sampling.json': 'settings',               # which records a fit samples and how
    'fixed-normalization.json': 'settings',        # the normalisation scales the model applies
    'hybrid-method.json': 'settings',              # which hybrid method variant the run used
    'initial-context.json': 'settings',            # the parent entry and teacher bank a run starts from
    'mode-curriculum.json': 'settings',            # seed and pattern counts of a curriculum
    'routing.json': 'settings',                    # grid-size routing between two heads
    'spacing-method.json': 'settings',             # the spacing feature a run adds
    'step-scales.json': 'settings',                # train-only step statistics used as scales
    'coupled-asset-provenance.json': 'record',     # provenance notes about assembled assets
    'gate-bounds.json': 'record',                  # result of an offline bound probe
    'gate-bounds.source.json': 'record',           # checksums of the probe's source and result
    'gradient-reference.json': 'record',           # reference values and checksums to check a fit against
    'point-initialization.json': 'record',         # checksum and run ids of the weights a component starts from
    'qualification.json': 'record',                # qualification status flags of a run
    'receipt.json': 'record',                      # export receipt with file checksums
    'request-initialization.json': 'record',       # checksum and run ids of starting weights
    'source-witness.json': 'record',               # witness of the sources and inputs a run was assembled from
    'viscosity-initialization.json': 'record',     # checksum and run ids of starting weights
}
DATA_SUFFIXES = ('.npz', '.npy', '.zip', '.pt', '.pth')


class CodePaths:
    """Maps the paths earlier runs executed (staged copies under runs/, which Git ignores) to files in Git.

    A path becomes a tracked file with identical content anywhere in the repository except the old
    project folder and earlier-run-code/. When several tracked files match, the one under
    experiments/<the run's study>/ is preferred, then any under experiments/, then the first in path
    order. Python code and settings files with no tracked twin are copied to earlier-run-code/: one
    copy per distinct content, at the first of its paths in sorted order with runs/ removed.
    Files the old Tuldok's run wiring needed (its runtime bootstrap and manifest, custody validator,
    extension SDK, contribution and extension descriptors, project document) and record files (see
    JSON_FILES) are left out. Archives and arrays are generated data and are not copied.
    """

    def __init__(self, root, exclude):
        self.root = Path(root)
        listing = subprocess.run(['git', 'ls-files', '-s', '-z'], cwd=self.root, check=True,
                                 capture_output=True).stdout.decode('utf-8')
        self.tracked, self.by_blob = set(), collections.defaultdict(list)
        excluded = [prefix.rstrip('/') + '/' for prefix in (exclude, EARLIER_RUN_CODE) if prefix]
        for record in filter(None, listing.split('\0')):
            meta, path = record.split('\t', 1)
            if any(path.startswith(prefix) for prefix in excluded):
                continue
            self.tracked.add(path)
            self.by_blob[meta.split()[1]].append(path)
        self.counts = collections.Counter()
        self.home = {}          # untracked path -> its copy under earlier-run-code/
        self.copies = {}        # path under earlier-run-code/ -> bytes

    def generated(self, path):
        parts = Path(path).parts
        name = parts[-1]
        if {'runtime-bootstrap', 'tuldok_extension_sdk'} & set(parts[:-1]) or name == 'runtime-manifest.json':
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

    @staticmethod
    def file_kind(path):
        """'code', 'settings', 'record' or 'data' for a file with no tracked copy."""
        name = Path(path).name
        if name.endswith('.py'):
            return 'code'
        if name.endswith(DATA_SUFFIXES):
            return 'data'
        if name in JSON_FILES:
            return JSON_FILES[name]
        raise ValueError(f'{path}: add {name!r} to JSON_FILES as settings or record.')

    def twin(self, path):
        twins = self.by_blob.get(git_blob_id((self.root / path).read_bytes()), [])
        study = study_of(path)
        return sorted(twins, key=lambda twin: (not (study and twin.startswith(f'experiments/{study}/')),
                                              not twin.startswith('experiments/'), twin))[:1]

    def prepare(self, paths):
        """Choose the earlier-run-code/ copy of every untracked code or settings file among `paths`."""
        by_content = collections.defaultdict(list)
        for path in sorted(set(paths)):
            if path in self.tracked or self.generated(path) or not (self.root / path).is_file():
                continue
            if not self.twin(path) and self.file_kind(path) in ('code', 'settings'):
                by_content[git_blob_id((self.root / path).read_bytes())].append(path)
        for same in by_content.values():
            first = same[0]
            home = f"{EARLIER_RUN_CODE}/{first[len('runs/'):] if first.startswith('runs/') else first}"
            self.copies[home] = (self.root / first).read_bytes()
            for path in same:
                self.home[path] = home

    def resolve(self, path):
        """(state, the path to write or None). States: tracked, rewritten, copied, generated, record,
        data (an archive or array with no copy in Git) and missing (not on this machine)."""
        if path in self.tracked:
            result = 'tracked', path
        elif self.generated(path):
            result = 'generated', None
        elif not (self.root / path).is_file():
            result = 'missing', path
        elif self.twin(path):
            result = 'rewritten', self.twin(path)[0]
        elif path in self.home:
            result = 'copied', self.home[path]
        else:
            kind = self.file_kind(path)
            result = (kind, None) if kind == 'record' else (kind, path)
            if kind in ('code', 'settings'):
                raise ValueError(f'{path} was not prepared for copying.')
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
        missing = [path for path in named if self.code_paths.resolve(path)[0] not in ('tracked', 'rewritten', 'copied')]
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
        self.cards = folder.parent / 'fits'
        self.card_paragraphs = collections.Counter()  # Purpose, Method, none, no card file, shortened
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

    def described(self, stem, text):
        """text led by the card's Purpose (else Method) paragraph, cut at a sentence end to keep within 4000."""
        card = self.cards / f'{stem}.md'
        label, lead = card_paragraph(card.read_text(encoding='utf-8')) if card.is_file() else ('no card file', '')
        self.card_paragraphs[label or 'none'] += 1
        short = sentences_within(lead, 4000 - len(text) - 1)
        if short != lead:
            self.card_paragraphs['shortened'] += 1
        return f'{short} {text}'[:4000] if short else text[:4000]

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
                              description=self.described(stem, f"Earlier model. {recipe.get('note', '')} Its run "
                                                                 f"folder was {recipe.get('run_folder')}."),
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

        # Every script and source path, mapped to a file in Git: a tracked twin or a copy in earlier-run-code/.
        entry_state, entrypoint = self.code_paths.resolve(recipe['entrypoint'])
        self.training_scripts[stem] = (entry_state, recipe['entrypoint'])
        written, absent, data, dropped, records = {}, [], [], [], []
        for source in recipe['sources']:
            if source == recipe.get('settings_file'):
                self.code_paths.counts['carried as settings'] += 1
                continue
            state, path = self.code_paths.resolve(source)
            if state == 'generated':
                dropped.append(source)
            elif state == 'record':
                records.append(source)
            else:
                written[source] = path
                if state == 'missing':
                    absent.append(source)
                elif state == 'data':
                    data.append(source)

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
                    supplied.append(f'{token} (a record file, not code or settings, so it is not in Git)'
                                    if found[0] in records else f'{token} (written by the old Tuldok for the run)')
            elif token.startswith('<'):
                self.unknown_tokens[token] += 1
                command.append(escape(token))
                supplied.append(f'{token} ({UNKNOWN_TOKEN_NOTE[token]})')
            else:
                command.append(escape(token))
        if len(command) > 64:
            raise ValueError(f'{stem}: the command has more than 64 words.')

        self.missing_sources[stem] = absent + data
        problems = []
        if entry_state == 'missing':
            problems.append(f"its training script {recipe['entrypoint']} is neither in Git nor on this machine")
        if absent:
            problems.append(f"{len(absent)} of its source files {'is' if len(absent) == 1 else 'are'} neither in Git "
                            f'nor on this machine (for example {absent[0]})')
        if lost:
            problems.append(f'it starts from weights at {lost[0]} whose producing model is not identified')
        if data and not problems:
            label = earlier('no', 'Cannot be rebuilt: the only missing pieces are archives or arrays (generated '
                            f"data) with no copy in Git: {', '.join(data)}.")
        elif data:
            problems.append(f"{len(data)} archive or array file{'s' if len(data) > 1 else ''} it reads "
                            f'{"has" if len(data) == 1 else "have"} no copy in Git (for example {data[0]})')
        if problems:
            label = earlier('no', 'Cannot be rebuilt: ' + '; '.join(problems) + '.')
        elif not data and supplied:
            label = earlier('needs-work', 'The recorded command takes arguments the old Tuldok supplied: '
                            + '; '.join(dict.fromkeys(supplied)) + '. They have to be replaced before it can run.')
        elif not data:
            label = earlier('after-inputs' if weights else 'yes')
        if dropped:
            notes.append(f'{len(dropped)} source files the old Tuldok generated for the run were left out.')
        if records:
            notes.append(f"{len(records)} record files (not code or settings) were left out: "
                         f"{', '.join(sorted({Path(r).name for r in records}))}.")

        run = {'command': command, 'environment': ENVIRONMENT_OF_RUNTIME[recipe['runtime']],
               'inputs': inputs, 'settings': recipe['settings'], 'outputs': self.outputs(recipe['outputs']),
               'resources': self.resources(recipe), 'sources': list(written.values())}
        run = {key: value for key, value in run.items() if value not in (None, {}, [])}
        description = (f"Earlier model, trained {recipe['trained_at'][:10]} by Tuldok run {recipe['tuldok_run']} "
                       f"({recipe['run_name']}). Converted from its extracted training recipe. The run executed "
                       'staged copies under runs/; each script and source is given as the tracked file with the '
                       'same content, or as its copy in earlier-run-code/.')
        if notes:
            description += ' ' + ' '.join(dict.fromkeys(notes))
        return definition('recipe', recipe['model'], description=self.described(stem, description), card=card,
                          labels=labels,
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


def case_values(observed, group_by, where):
    """The tuldok_results_v1 `values` of one subject, sorted, with groups limited to `group_by`.

    A dimension recorded as null is not a group value, so that value carries no group for it.
    """
    values, seen = [], {}
    for case, metric, value, reason, texts, source in observed:
        group = {key: texts[key] for key in group_by if key in texts and texts[key] != 'null'}
        identity = (case, metric, tuple(sorted(group.items())))
        if identity in seen:
            raise ValueError(f'{where}: case {case!r}, metric {metric!r}, group {dict(identity[2])} has two values, '
                             f'from result sets {seen[identity]!r} and {source!r}.')
        seen[identity] = source
        entry = {'case': case, 'metric': metric, 'value': value}
        if value is None:
            entry['reason'] = reason
        if group:
            entry['group'] = group
        values.append(entry)
    return sorted(values, key=lambda e: (e['metric'], e['case'], sorted(e.get('group', {}).items())))


def summarise_values(values, summarise):
    """{(metric, group): summary} for one subject: the ungrouped row and one row per single key=value."""
    measured, missing = collections.defaultdict(list), collections.Counter()
    for entry in values:
        for group in ['', *(f'{key}={text}' for key, text in sorted(entry.get('group', {}).items()))]:
            if entry['value'] is None:
                missing[entry['metric'], group] += 1
            else:
                measured[entry['metric'], group].append(entry['value'])
    return {key: summarise(measured[key], missing[key]) for key in sorted(set(measured) | set(missing))}


def dump_values(document):
    """A values file: its header fields, then one value per line, so large files stay readable."""
    head = {key: value for key, value in document.items() if key != 'values'}
    lines = [json.dumps(entry, ensure_ascii=False, allow_nan=False) for entry in document['values']]
    text = json.dumps(head, indent=1, ensure_ascii=False)[:-2] + ',\n "values": [\n  ' + ',\n  '.join(lines) + '\n ]\n}\n'
    return text.encode('utf-8')


class Benchmarks:
    """The project's measurement_source result sets, grouped into benchmark definitions and summarised."""

    def __init__(self, source, matcher):
        self.source = source
        self.matcher = matcher
        blobs = stored_files(source)
        inventory = load(source / '.tuldok' / 'project.json')['inventory']
        self.result_sets = []
        for entry in sorted((e for e in inventory if e.get('family') == 'measurement_source'), key=lambda e: e['path']):
            record = load(source / '.tuldok' / entry['path'])
            found = [path for name, path in blobs.items() if name.startswith(record['blob']['sha256'])]
            if len(found) != 1:
                raise FileNotFoundError(f"Result set {record['title']!r}: {len(found)} stored files match its checksum.")
            self.result_sets.append({'title': record['title'], 'imported_at': record['imported_at'], 'file': found[0]})
        self.matches = []   # (benchmark, subject id, recipe reference or None, route or reason)
        self.values = {}    # (benchmark, subject name) -> its tuldok_results_v1 document

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
        observed = collections.defaultdict(list)   # subject id -> (case, metric, value, reason, dimensions, set)
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
                subjects.setdefault(subject['id'], subject)
                produced[subject['id']] = max(produced.get(subject['id'], ''), member['imported_at'][:10])
            cases.update(case['id'] for case in document['cases'])
            for observation in document['observations']:
                dimensions = observation.get('dimensions') or {}
                if isinstance(dimensions, list):
                    dimensions = {entry['id']: entry['value'] for entry in dimensions}
                texts = {key: dimension_text(value) for key, value in dimensions.items()}
                for key, text in texts.items():
                    dimension_values[key].add(text)
                measured = observation['state'] == 'measured' and observation['value'] is not None
                reason = None if measured else (' '.join(str(observation.get('reason') or '').split())[:400]
                                                or 'Recorded as unavailable by the earlier study harness.')
                observed[observation['subject_id']].append(
                    (observation['case_id'], observation['metric_id'].replace('-', '_'),
                     observation['value'] if measured else None, reason, texts, member['title']))
            del document
        if len(scopes) != 1:
            raise ValueError(f'The result sets grouped as {title!r} have different scopes: {sorted(scopes)}.')
        (scope_id, scope_label, population), = scopes
        # Group by the per-value dimensions that actually split the values.
        group_by = sorted(key for key, texts in dimension_values.items() if len(texts) > 1)
        for key in group_by:
            if not METRIC_ID.fullmatch(key):
                raise ValueError(f'Dimension {key!r} of {title!r} cannot be a group_by key.')

        # Each subject's per-case values, the same list that --values-output writes, and the summary rows
        # computed from exactly that list.
        summary_rows = []
        for identifier in sorted(observed):
            values = case_values(observed[identifier], group_by, f'{title!r}, subject {identifier!r}')
            subject = subject_name(identifier)
            self.values[name, subject] = {'format': 'tuldok_results_v1', 'benchmark': name, 'subject': subject,
                                          'commit': '', 'produced': produced[identifier], 'values': values}
            for (metric, group), summary in summarise_values(values, summarise).items():
                summary_rows.append(summary_row(subject=subject, metric=metric, group=group, commit='',
                                                produced=produced[identifier], **summary))

        subject_entries, names = [], set()
        for identifier in sorted(subjects):
            subject = subjects[identifier]
            entry = {'name': subject_name(identifier)}
            if entry['name'] in names:
                raise ValueError(f'Two subjects of {title!r} become the name {entry["name"]!r}.')
            names.add(entry['name'])
            label = subject.get('label') or identifier
            # Many labels already begin with the id ("C16 · GPU pressure request"); do not repeat it.
            entry['title'] = (label if label.startswith(identifier) else f'{identifier} · {label}')[:200]
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
                       + f'. Summaries are in results/{name}.csv. The per-case values are not in the folder; '
                       'scripts/build_tuldok_project.py --values-output writes them as tuldok_results_v1 files.')
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
        f"{len(portfolio['relationships'])} authored comparisons between fits with excerpts. The groups are "
        "carried as tuldok.json's groups and each recipe's group, the comparisons as compared_with on the "
        'recipes; the model entries (aliases, review state, summaries that partly repeat the cards) are not.',
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


def build(source, output, tuldok, seed=SEED_DEFAULT):
    """Everything the script owns, as {relative path: bytes}, plus what the report needs."""
    sys.path.insert(0, str(Path(tuldok).resolve()))
    from tuldok_project.results import SummaryRow, summarise, write_results

    notes = source / 'notes'
    code_root = (Path(output) / PROJECT['code_root']).resolve()
    old_project = Path(source).resolve().relative_to(code_root).as_posix()
    code_paths = CodePaths(code_root, exclude=old_project)
    training_sets = TrainingSets(load(notes / 'recipes' / 'training-sets.json'), code_paths)
    recipes = Recipes(notes / 'recipes', code_paths)
    code_paths.prepare(path for recipe in recipes.recorded.values() if recipe.get('command')
                       for path in [recipe['entrypoint'], *recipe['sources']] if path != recipe.get('settings_file'))
    benchmarks = Benchmarks(source, SubjectMatcher(notes, recipes.stem_of_fit))

    documents = {**training_sets.definitions(), **recipes.definitions()}
    groups, fields, map_lines = map_fields(load(notes / '00-models.portfolio.json'), load(seed), recipes.stem_of_fit)
    add_map_fields(documents, fields)
    files = {path: dump(document) for path, document in documents.items()}
    for path, content in benchmarks.build(summarise, write_results, SummaryRow).items():
        files[path] = content if isinstance(content, bytes) else dump(content)
    files['tuldok.json'] = dump({**PROJECT, 'groups': groups, 'map': {'chart': MAP_CHART}})
    for path in sorted([*notes.glob('*.md'), *(notes / 'fits').glob('*.md')]):
        files[path.relative_to(source).as_posix()] = path.read_bytes()

    hand_written = {path for path, data in existing_files(output).items() if not owned_by_script(path, data)}
    clash = sorted(set(files) & hand_written)
    if clash:
        raise ValueError(f'These files are written by hand and the script would overwrite them: {clash}')
    run_code = {path[len(EARLIER_RUN_CODE) + 1:]: data for path, data in code_paths.copies.items()}
    run_code['README.md'] = EARLIER_RUN_CODE_README.encode('utf-8')
    return files, run_code, {'benchmarks': benchmarks, 'recipes': recipes, 'training_sets': training_sets,
                             'notes': notes, 'code_paths': code_paths, 'code_root': code_root,
                             'map': (groups, fields, map_lines)}


EARLIER_RUN_CODE_README = '''# Earlier run code

The code and settings files that the lab's earlier training runs executed, in the versions they ran.
Those runs worked from staged copies under `runs/`, which Git ignores, and these versions were never
committed anywhere else. They are kept here so the earlier recipes in `projects/surrogate-lab/recipes/`
name files that are in Git.

A file from `runs/<path>` is here at `<path>`. Each distinct file is here once, at the first of its
paths in sorted order; every recipe that ran an identical file points at that copy. Archives, arrays,
weights and record files (checksums, witnesses, receipts, qualification records) are not copied.

`scripts/build_tuldok_project.py` writes this folder and removes anything it no longer needs. Do not
edit it by hand.
'''


def map_fields(portfolio, seed, stem_of_fit):
    """The project's groups and, per recipe stem, the map fields of the earlier models, plus lines for the report.

    Groups and comparisons are the authored ones of notes/00-models.portfolio.json; change, derived_from
    and the label short come from the seed (scripts/earlier-model-map.json), keyed by fit id. A field
    that would name something that is not a recipe of the folder is left out and reported.
    """
    def recipe(fit):
        return f'recipes/{stem_of_fit[fit]}' if fit in stem_of_fit else None

    groups, lines = [], []
    for group in portfolio['groups']:
        if not GROUP_ID.fullmatch(group['id']) or len(group['label']) > 200 or len(group['description'] or '') > 2000:
            raise ValueError(f"Group {group['id']!r} cannot be written as a project group as it is.")
        entry = {'id': group['id'], 'title': group['label'], 'parent': group['parent'],
                 'description': group['description']}
        groups.append({key: value for key, value in entry.items() if value})
    known = {group['id'] for group in groups}
    fields = collections.defaultdict(dict)
    for model in portfolio['models']:
        if model['group_id'] not in known:
            raise ValueError(f"Model {model['id']} is in the unknown group {model['group_id']!r}.")
        if recipe(model['id']):
            fields[stem_of_fit[model['id']]]['group'] = model['group_id']
        else:
            lines.append(f"group of {model['id']} left out: it has no recipe")

    for item in portfolio['relationships']:
        model, other = recipe(item['from']), recipe(item['to'])
        if not model or not other or model == other:
            lines.append(f"comparison {item['from']} -> {item['to']} left out: "
                         + ('it compares a model with itself' if model and model == other else 'a side has no recipe'))
            continue
        if not item['explanation'] or len(item['explanation']) > 2000 or len(item.get('excerpt') or '') > 2000:
            raise ValueError(f"Comparison {item['from']} -> {item['to']}: its texts do not fit the format.")
        compared = fields[model[len('recipes/'):]].setdefault('compared_with', [])
        same = [entry for entry in compared if entry['recipe'] == other]
        if same:
            # The format allows one entry per other recipe: keep the first, with both explanations.
            if len(same[0]['explanation']) + 1 + len(item['explanation']) > 2000 or \
                    (item.get('excerpt') and same[0].get('excerpt', item['excerpt']) != item['excerpt']):
                raise ValueError(f"Comparisons {item['from']} -> {item['to']} cannot be joined into one.")
            same[0]['explanation'] += ' ' + item['explanation']
            if item.get('excerpt'):
                same[0]['excerpt'] = item['excerpt']
            lines.append(f"comparison {item['from']} -> {item['to']} given twice: joined into the first")
            continue
        compared.append({'recipe': other, 'explanation': item['explanation'],
                         **({'excerpt': item['excerpt']} if item.get('excerpt') else {})})

    models = seed['models']
    for fit, entry in sorted(models.items()):
        if fit not in stem_of_fit:
            lines.append(f'seed entry {fit} left out: it has no recipe')
            continue
        own = fields[stem_of_fit[fit]]
        own['change'] = entry['change']
        if entry.get('short'):
            own['short'] = entry['short']
        if entry.get('derived_from') and recipe(entry['derived_from']) and entry['derived_from'] != fit:
            own['derived_from'] = recipe(entry['derived_from'])
        elif entry.get('derived_from'):
            lines.append(f"derived_from of {fit} left out: {entry['derived_from']} is not another recipe")
    lines += [f'recipe of {fit} has no seed entry' for fit in sorted(set(stem_of_fit) - set(models))]
    return groups, fields, lines


def add_map_fields(documents, fields):
    """Put group, change, derived_from, compared_with and the label short on the recipe documents."""
    for stem, own in fields.items():
        document = documents[f'recipes/{stem}.json']
        for key in ('group', 'change', 'derived_from', 'compared_with'):
            if key in own:
                document[key] = own[key]
        if 'short' in own:
            document['labels']['short'] = own['short']
            if len(document['labels']) > 32:
                raise ValueError(f'{stem}: more than 32 labels.')


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


def report(files, output, details, run_code_folder):
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
                 f"content, {counts['copied']} pointed at a copy in {EARLIER_RUN_CODE}/, {counts['tracked']} already "
                 f"tracked, {counts['generated']} generated by the old Tuldok and left out, {counts['record']} record "
                 f"files left out, {counts['data']} archives or arrays with no copy in Git, {counts['missing']} not on "
                 f"this machine, {counts['carried as settings']} settings files carried as the recipe's settings.")
    lines.append('Untracked JSON files, as classified in JSON_FILES:')
    for kind in ('settings', 'record'):
        lines.append(f"  {kind}: " + ', '.join(name for name, k in sorted(JSON_FILES.items()) if k == kind))
    scripts = details['recipes'].training_scripts
    by_state = collections.Counter(state for state, _ in scripts.values())
    lines.append(f"Training scripts of the {len(scripts)} earlier recipes with a command: "
                 + ', '.join(f'{count} {state}' for state, count in sorted(by_state.items())) + '.')
    absent = collections.defaultdict(list)
    for stem, (state, path) in sorted(scripts.items()):
        if state == 'missing':
            absent[study_of(path) or 'not under runs/'].append(stem)
    for study, stems in sorted(absent.items()):
        lines.append(f'  not in Git or on this machine, {study} ({len(stems)}):')
        lines += [f'    {stem}' for stem in stems]
    lines.append("Earlier recipes' descriptions led by a card paragraph: "
                 + ', '.join(f'{name} {count}' for name, count in sorted(details['recipes'].card_paragraphs.items()))
                 + '.')
    missing = {stem: paths for stem, paths in details['recipes'].missing_sources.items() if paths}
    names = collections.Counter(Path(p).name for paths in missing.values() for p in set(paths))
    lines.append(f'Earlier recipes with source files still not in Git: {len(missing)}. Their names: '
                 + ', '.join(f'{name} ({count})' for name, count in sorted(names.items())) + '.')
    run_code = existing_files(run_code_folder)
    lines.append(f'{EARLIER_RUN_CODE}/: {sum(map(len, run_code.values())) / 1e6:.2f} MB in {len(run_code)} files.')
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
    rows = {path[len('results/'):-len('.csv')]: data.count(b'\n') - 1
            for path, data in sorted(files.items()) if path.startswith('results/')}
    lines.append(f'Summary rows: {sum(rows.values())} in {len(rows)} results files.')
    lines += [f'  {name}: {count}' for name, count in rows.items()]
    size = sum(map(len, existing_files(output).values()))
    lines.append(f'Folder size: {size / 1e6:.2f} MB in {len(existing)} files '
                 f'(notes {notes_size / 1e6:.2f} MB, results {results / 1e6:.2f} MB).')
    groups, fields, map_lines = details['map']
    own = list(fields.values())
    changes = collections.Counter(entry['change'] for entry in own if 'change' in entry)
    parents = collections.Counter(entry['derived_from'] for entry in own if 'derived_from' in entry)
    lines.append(f"Map: {len(groups)} groups; recipes with a group {sum('group' in e for e in own)}, with a change "
                 f"{sum(changes.values())} (" + ', '.join(f'{k} {v}' for k, v in sorted(changes.items())) + '), '
                 f"derived from another {sum(parents.values())} ("
                 + ', '.join(f'{k} {v}' for k, v in sorted(parents.items())) + f"), with the label short "
                 f"{sum('short' in e for e in own)}, with comparisons {sum('compared_with' in e for e in own)} "
                 f"({sum(len(e.get('compared_with', [])) for e in own)} comparisons). Track: {MAP_CHART}"
                 + ('' if (Path(output) / f'{MAP_CHART}.json').is_file() else ', which is not in the folder') + '.')
    lines += [f'  {line}' for line in map_lines]
    lines.append('Left in the old notes and not carried:')
    lines += [f'  - {line}' for line in notes_left_out(details['notes'])]
    return '\n'.join(lines)


VALUES_DEFAULT = Path('runs/tuldok-results')


def write_values(folder, documents):
    """Write <folder>/<benchmark>/<subject>.json for every document, remove stale ones there, and say what was written."""
    wanted = {f'{benchmark}/{subject}.json': document for (benchmark, subject), document in documents.items()}
    for benchmark in sorted({benchmark for benchmark, _ in documents}):
        for path in sorted((folder / benchmark).glob('*.json')):
            if f'{benchmark}/{path.name}' not in wanted:
                path.unlink()
    written = 0
    for path, document in sorted(wanted.items()):
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = dump_values(document)
        if not target.is_file() or target.read_bytes() != data:
            target.write_bytes(data)
            written += 1
    count = sum(len(document['values']) for document in documents.values())
    size = sum((folder / path).stat().st_size for path in wanted)
    return (f'Values files in {folder}: {len(wanted)} files ({written} rewritten), {count} values, '
            f'{size / 1e6:.1f} MB.')


# ---------------------------------------------------------------- the earlier products, as an adoption manifest

ADOPTION_DEFAULT = Path('runs/tuldok-adoption/surrogate-lab.json')
RUN_CODE_PAYLOADS = ('source', 'host-sources', 'runtime-bootstrap')
CHECKSUM = re.compile(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')


def old_runs(source):
    """{run id: state, start, artifacts and payloads} from the old folder's run_history records, read one at a time.

    `artifacts` are the run's outputs as recorded ({name, role, bytes, sha256}); `payloads` are
    (name, role, checksum, bytes) of everything kept with the run: inputs, staged code, custody proofs, outputs.
    """
    runs = {}
    for path in sorted((Path(source) / '.tuldok' / 'records' / 'run_history').rglob('*.json')):
        record = load(path)
        run = record['run']
        if run['id'] in runs:
            raise ValueError(f"Run {run['id']} has two run_history records.")
        runs[run['id']] = {'state': run['state'], 'created_at': run['created_at'], 'artifacts': run['artifacts'],
                           'payloads': [(p['name'], p['role'], p['blob']['sha256'], p['blob']['bytes'])
                                        for p in record['payloads']]}
    return runs


def gigabytes(count):
    return f'{count / 1e9:.3f} GB'


def training_set_folder(entry, path):
    """The deepest folder of a training-sets.json entry that holds the recorded `path`, or None.

    Its folders are `folders` (relative to the code root, or '' for the base of paths recorded without one)
    and, for an entry with `folder_pattern`, every directory above `path` that the pattern matches.
    """
    found = [folder for folder in entry.get('folders') or [] if folder == '' or path.startswith(folder + '/')]
    if entry.get('folder_pattern'):
        found += [parent for parent in (p.as_posix() for p in PurePosixPath(path).parents)
                  if re.search(entry['folder_pattern'], parent)]
    return max(found, key=len, default=None)


def common_directory(folders):
    """The deepest directory that holds all `folders` ('' when they share none)."""
    common = []
    for level in zip(*(folder.split('/') if folder else [] for folder in folders)):
        if len(set(level)) > 1:
            break
        common.append(level[0])
    return '/'.join(common)


def dataset_key(entry, path, folders):
    """The path of a recorded input file inside its dataset's directory, or None when it cannot be placed.

    The dataset's directory is its one folder, or the deepest directory holding all its folders (the
    listed ones, else those the folder_pattern matched), so files of different folders never share a key.
    """
    directory = common_directory(folders)
    if training_set_folder(entry, path) is None or not (directory == '' or path.startswith(directory + '/')):
        return None
    key = path[len(directory) + 1:] if directory else path
    parts = key.split('/')
    if path.startswith('/') or any(part in ('', '.', '..') for part in parts):
        return None
    return key


def examples(items):
    return f" (for example {'; '.join(items[:3])})" if items else ''


def adoption(source, files, recorded, training_sets):
    """The tuldok_adoption_v1 manifest of the earlier recipes and datasets this script writes, and lines reporting it.

    A recipe's entry keys the files of its old run (its labels.tuldok_run) that are in artifacts/sha256/
    by its declared output paths. A declared output has a file when the old run recorded an artifact at
    that path and artifacts/sha256/ holds it as a regular file of the recorded size.

    A dataset declares no outputs, so its product is its directory as a whole. Its entry holds the files
    the earlier recipes' runs read from it (their inputs of kind 'training data', as notes/recipes/<stem>.json
    records them, retained with the run as input/<name>), keyed by dataset_key. A path that two different
    stored files claim is left out, as is a file that cannot be placed.
    `recorded` is the old notes/recipes/<stem>.json documents; `training_sets` the training-sets.json entries by id.
    """
    blobs = stored_files(source)
    stats = {checksum: path.lstat() for checksum, path in blobs.items()}
    runs = old_runs(source)

    def usable(checksum, size):
        stored = stats.get(checksum)
        return stored is not None and stat.S_ISREG(stored.st_mode) and stored.st_size == size

    products, named, coverage, mismatches = [], collections.defaultdict(list), collections.Counter(), \
        collections.defaultdict(list)
    definitions = {path[:-len('.json')]: json.loads(files[path]) for path in sorted(files)
                   if path.split('/', 1)[0] in ('recipes', 'datasets') and path.endswith('.json')}
    for reference, document in definitions.items():
        if not reference.startswith('recipes/'):
            continue
        declared = sorted(output['path'] for output in document.get('run', {}).get('outputs', {}).values())
        run_id = document.get('labels', {}).get('tuldok_run')
        if run_id and run_id not in runs:
            mismatches['old run with no run_history record'].append(f'{reference}: run {run_id}')
        run = runs.get(run_id, {'artifacts': []})
        artifacts = {artifact['name']: artifact for artifact in run['artifacts']}
        mismatches['old artifact with no declared output'] += [f'{reference}: {name}'
                                                               for name in sorted(set(artifacts) - set(declared))]
        found = {}
        for name in declared:
            artifact = artifacts.get(name)
            if artifact is None:
                mismatches['declared output with no old artifact'].append(f'{reference}: {name}')
            elif artifact['sha256'] not in stats:
                mismatches['old artifact not in artifacts/sha256'].append(f'{reference}: {name}')
            elif not usable(artifact['sha256'], artifact['bytes']):
                mismatches['stored file not a regular file of the recorded size'].append(f'{reference}: {name}')
            else:
                found[name] = artifact['sha256']
        coverage['no declared outputs' if not declared else 'every' if len(found) == len(declared)
                 else 'some' if found else 'none'] += 1
        if found:
            if len(found) > 200:
                raise ValueError(f'{reference}: {len(found)} files, more than one entry may hold.')
            products.append({'reference': reference, 'note': f"Tuldok run {run_id} of {run['created_at'][:10]}",
                             'files': {name: str(blobs[checksum].resolve()) for name, checksum in found.items()}})
            named['recipes'] += found.values()

    # The files each dataset's readers took from it: (run, recorded path, checksum or None, recorded bytes).
    reads = collections.defaultdict(list)
    for recipe in recorded.values():
        run = runs.get(recipe.get('tuldok_run'))
        if run is None:
            continue
        retained = {name[len('input/'):]: (checksum, size) for name, _, checksum, size in run['payloads']
                    if name.startswith('input/')}
        for item in recipe['inputs']:
            if item.get('kind') == 'training data':
                checksum, size = retained.get(item['name'], (None, item.get('bytes')))
                reads[f"datasets/{item['training_set']}"].append((recipe['tuldok_run'], item['path'], checksum, size))
    datasets, problems, directories = collections.Counter(), collections.defaultdict(list), collections.Counter()
    conflicts, unplaced, too_many = collections.defaultdict(set), set(), []
    for reference, document in definitions.items():
        if not reference.startswith('datasets/'):
            continue
        if document.get('run', {}).get('outputs'):
            raise ValueError(f'{reference} declares outputs; the converter has no rule for adopting them.')
        entry, read = training_sets.get(reference.split('/', 1)[1]) or {}, reads.get(reference, [])
        if not read:
            datasets['read by no earlier recipe'] += 1
            continue
        folders = entry.get('folders') or sorted({training_set_folder(entry, path) for _, path, _, _ in read} - {None})
        directories['its one folder' if len(folders) == 1 else 'the directory holding its listed folders'
                    if entry.get('folders') else 'the directory holding the folders its folder_pattern matched'] += 1
        claims = collections.defaultdict(set)
        for run_id, path, checksum, size in read:
            key = dataset_key(entry, path, folders) if folders else None
            if checksum is None:
                problems['read file not retained with its run'].append(f'{reference}: {path}')
            elif not usable(checksum, size):
                problems['read file not in artifacts/sha256 as a regular file of the recorded size'].append(
                    f'{reference}: {path}')
            elif key is None:
                problems['read file that cannot be placed in its dataset'].append(f'{reference}: {path}')
                unplaced.add(checksum)
            else:
                claims[key].add(checksum)
        for key, checksums in sorted(claims.items()):
            if len(checksums) > 1:
                conflicts[reference, key] = checksums
        found = {key: checksums.pop() for key, checksums in sorted(claims.items()) if len(checksums) == 1}
        if not found:
            datasets['every read file left out'] += 1
        elif len(found) > 200:
            too_many.append(f'{reference} ({len(found)} files)')
            datasets['more than 200 files'] += 1
        else:
            datasets['with an entry'] += 1
            runs_reading = len({run_id for run_id, _, _, _ in read})
            products.append({'reference': reference,
                             'note': f"The files that {runs_reading} earlier Tuldok run{'s' * (runs_reading != 1)} "
                                     'read from this dataset.',
                             'files': {key: str(blobs[checksum].resolve()) for key, checksum in found.items()}})
            named['datasets'] += found.values()
    if len(products) > 5000:
        raise ValueError(f'{len(products)} entries, more than a manifest may hold.')
    every = [checksum for kind in named for checksum in named[kind]]

    # What refers to each stored file no entry names: the run records first, then every other record family.
    inputs_of_run = {recipe['tuldok_run']: {item['name']: item for item in recipe['inputs']}
                     for recipe in recorded.values() if recipe.get('tuldok_run')}
    refers = {}
    # The runs of earlier recipes first, so a file such a run read is described by what its recipe says it is.
    ordered = sorted(runs.items(), key=lambda item: (item[0] not in inputs_of_run, item[0]))
    for run_id, run in ordered:
        whose = 'a recipe' if run_id in inputs_of_run else 'no recipe'
        for artifact in run['artifacts']:
            refers.setdefault(artifact['sha256'], f"output ({artifact['role']}) of an old run with {whose}, "
                                                  f"{run['state']}")
    for run_id, run in ordered:
        inputs = inputs_of_run.get(run_id)
        for name, role, checksum, _ in run['payloads']:
            group, _, rest = name.partition('/')
            if group == 'input' and inputs is None:
                label = 'input of an old run with no recipe'
            elif group == 'input':
                label = f"input of an earlier recipe's run: {inputs.get(rest, {}).get('kind') or 'not in the recipe'}"
            elif group in RUN_CODE_PAYLOADS:
                label = f'code staged for an old run ({group}/)'
            else:
                label = f'{role} file of an old run'
            refers.setdefault(checksum, label)
    unnamed = set(blobs) - set(every)
    left = unnamed - set(refers)
    records = Path(source) / '.tuldok' / 'records'
    families = sorted((p.name for p in records.iterdir() if p.is_dir() and p.name != 'run_history'),
                      key=lambda family: (family == 'project_artifact', family))   # the bare inventory last
    for family in families:
        for path in sorted((records / family).rglob('*.json')) if left else []:
            for checksum in left & set(CHECKSUM.findall(path.read_text(encoding='utf-8'))):
                refers[checksum] = f'named in {family} records'
            left -= set(refers)
    by_label = collections.defaultdict(list)
    for checksum in unnamed:
        by_label[refers.get(checksum, 'named by no record')].append(stats[checksum].st_size)

    def total(checksums):
        return sum(stats[checksum].st_size for checksum in checksums)

    lines = [f'Adoption manifest: {len(products)} entries naming {len(every)} files, {total(every)} bytes '
             f'({gigabytes(total(every))}); {len(set(every))} distinct stored files, {gigabytes(total(set(every)))}.']
    lines.append(f'  Earlier recipes: {sum(coverage.values())}; every declared output has a file '
                 f"{coverage['every']}, some {coverage['some']}, none {coverage['none']}, no declared outputs "
                 f"{coverage['no declared outputs']}. Their entries name {len(named['recipes'])} files, "
                 f"{gigabytes(total(named['recipes']))}.")
    for problem in ('declared output with no old artifact', 'old artifact with no declared output',
                    'old artifact not in artifacts/sha256', 'stored file not a regular file of the recorded size',
                    'old run with no run_history record'):
        lines.append(f'    {problem}: {len(mismatches[problem])}{examples(mismatches[problem])}')
    lines.append(f"  Earlier datasets: {sum(datasets.values())}, none declaring outputs; with an entry "
                 f"{datasets['with an entry']}, without {sum(datasets.values()) - datasets['with an entry']} (read by "
                 f"no earlier recipe {datasets['read by no earlier recipe']}, every read file left out "
                 f"{datasets['every read file left out']}, more than 200 files {datasets['more than 200 files']}"
                 f"{examples(too_many)}). Their entries name {len(named['datasets'])} files, "
                 f"{gigabytes(total(named['datasets']))}.")
    lines.append('    Keys are paths inside the dataset\'s directory, which is '
                 + ', '.join(f'{label} for {count}' for label, count in sorted(directories.items())) + '.')
    lines.append(f"    Reads of a dataset file by an earlier recipe's run: {sum(map(len, reads.values()))}.")
    for problem in ('read file not retained with its run',
                    'read file not in artifacts/sha256 as a regular file of the recorded size',
                    'read file that cannot be placed in its dataset'):
        lines.append(f'    {problem}: {len(problems[problem])} reads{examples(problems[problem])}')
    lines[-1] += f'; {len(unplaced)} stored files, {gigabytes(total(unplaced))}'
    conflicting = set().union(*conflicts.values())
    lines.append(f'    paths of a dataset that two or more different stored files claim, left out: {len(conflicts)} '
                 f'paths, {len(conflicting)} stored files, {gigabytes(total(conflicting))}'
                 + examples([f'{reference}: {key} ({len(checksums)} files)'
                             for (reference, key), checksums in sorted(conflicts.items())]))
    lines.append(f'Files in artifacts/sha256 that no entry names: {len(unnamed)}, {total(unnamed)} bytes '
                 f'({gigabytes(total(unnamed))}), by what refers to them:')
    lines += [f'  {len(sizes):5d} {gigabytes(sum(sizes)):>11s}  {label}'
              for label, sizes in sorted(by_label.items(), key=lambda item: (-sum(item[1]), item[0]))]
    return {'format': 'tuldok_adoption_v1', 'products': products}, lines


def write_adoption(target, source, files, recorded, training_sets):
    manifest, lines = adoption(source, files, recorded, training_sets)
    if manifest['products']:
        data = dump(manifest)
        if len(data) > 16 * 2 ** 20:
            raise ValueError(f'The manifest is {len(data)} bytes, more than 16 MiB.')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        lines.insert(0, f'Wrote {target.resolve()}.')
    else:
        lines.insert(0, f'No earlier definition has a file to adopt; {target} was not written.')
    return '\n'.join(lines)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--tuldok', required=True, type=Path, help='the Tuldok repository, to import tuldok_project')
    parser.add_argument('--source', type=Path, default=Path('projects/rheon-model-portfolio-unified-04'))
    parser.add_argument('--output', type=Path, default=Path('projects/surrogate-lab'))
    parser.add_argument('--check', action='store_true',
                        help='write nothing; fail if the folder differs from what the conversion would write '
                             '(the folder is no longer required to match)')
    parser.add_argument('--values-output', nargs='?', type=Path, const=VALUES_DEFAULT, metavar='DIR',
                        help='also write the per-case values of every earlier benchmark and subject as '
                             f'tuldok_results_v1 files under DIR (default {VALUES_DEFAULT}, from the code root); '
                             'they are generated data, outside the folder and not covered by --check')
    parser.add_argument('--adoption-output', nargs='?', type=Path, const=ADOPTION_DEFAULT, metavar='FILE',
                        help='also write the tuldok_adoption_v1 manifest that makes the files of the earlier runs, '
                             'as artifacts/sha256 of --source holds them, the products of the earlier recipes and '
                             f'datasets (default {ADOPTION_DEFAULT}, from the code root). It holds absolute paths, '
                             'so it stays outside Git; it is not covered by --check')
    parser.add_argument('--seed', type=Path, default=SEED_DEFAULT,
                        help='the change, derived_from and short label of each earlier model, by fit id '
                             '(default scripts/earlier-model-map.json)')
    options = parser.parse_args(arguments)

    files, run_code, details = build(options.source, options.output, options.tuldok, options.seed)
    run_code_folder = details['code_root'] / EARLIER_RUN_CODE
    folders = [(options.output, files, owned_by_script), (run_code_folder, run_code, lambda path, data: True)]
    plans = []
    for folder, wanted, owned in folders:
        existing = existing_files(folder)
        plans.append((folder, wanted,
                      sorted(path for path, data in wanted.items() if existing.get(path) != data),
                      sorted(path for path, data in existing.items() if owned(path, data) and path not in wanted)))

    if not options.check:
        for folder, wanted, changed, stale in plans:
            for path in stale:
                (folder / path).unlink()
            for path in changed:
                target = folder / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wanted[path])
            for directory in sorted((p for p in folder.rglob('*') if p.is_dir()), reverse=True):
                if not any(directory.iterdir()):
                    directory.rmdir()
    print(report(files, options.output, details, run_code_folder))
    if options.values_output is not None:
        folder = options.values_output
        if folder == VALUES_DEFAULT:
            folder = details['code_root'] / VALUES_DEFAULT
        print(write_values(folder, details['benchmarks'].values))
    if options.adoption_output is not None:
        target = options.adoption_output
        if target == ADOPTION_DEFAULT:
            target = details['code_root'] / ADOPTION_DEFAULT
        print(write_adoption(target, options.source, files, details['recipes'].recorded,
                             details['training_sets'].sets))
    if options.check:
        out_of_date = [f'{folder.name}/{path}' for folder, _, changed, stale in plans for path in changed + stale]
        if out_of_date:
            print(f'\nOut of date: {len(out_of_date)} files differ, are missing or should not be there.')
            for path in out_of_date[:20]:
                print(f'  {path}')
            return 1
        print('\nThe folder and earlier-run-code/ match what the script writes.')
        return 0
    print('\n' + '; '.join(f'{folder.name}: wrote {len(changed)} files, removed {len(stale)}'
                           for folder, _, changed, stale in plans) + '.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
