"""Stdlib construction helpers for the current explicit E2 context and D2 files.

The host owns admission, policy validation and publication. This module never
loads arrays/weights, computes metrics, authorizes a context or evaluates quality.
"""
import copy
import hashlib
from functools import wraps
import json
import math
import os
from pathlib import Path
import re
import stat

CONTEXT_FORMAT = 'extension_evidence_context_v1'
CONTEXT_LIMIT = 64 * 1024
JSON_LIMIT = 8 * 1024 * 1024
SAFE_INTEGER = 2**53 - 1
SHA = re.compile('[a-f0-9]{64}')
ID = re.compile('[a-f0-9]{32}')
NAME = re.compile('[A-Za-z0-9][A-Za-z0-9_.-]{0,119}')


class EvidenceError(ValueError):
    def __init__(self, message, state='invalid'):
        super().__init__(message)
        self.state = state


def require(condition, message, state='invalid'):
    if not condition:
        raise EvidenceError(message, state)


def boundary(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except EvidenceError:
            raise
        except OSError:
            raise EvidenceError('Required evidence file is unavailable.', 'unavailable') from None
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
            raise EvidenceError('Malformed evidence structure.') from None
    return call


def closed(value, keys, label):
    require(type(value) is dict and set(value) == set(keys), label + ' fields differ.')


def natural(value, label, minimum=0, maximum=SAFE_INTEGER):
    require(type(value) is int and minimum <= value <= maximum, label + ' must be an exact bounded integer.')
    return value


def sha(value):
    require(type(value) is str and SHA.fullmatch(value), 'Expected a lowercase SHA-256.')
    return value


def identifier(value):
    require(type(value) is str and ID.fullmatch(value), 'Expected a canonical identifier.')
    return value


def name(value):
    require(type(value) is str and NAME.fullmatch(value) and value not in ('custody-receipt.json', 'artifact-manifest.json'), 'Expected a flat artifact name.')
    return value


def _json(value, depth=0, budget=None):
    budget = [65536] if budget is None else budget
    budget[0] -= 1
    require(depth <= 24 and budget[0] >= 0, 'JSON exceeds the depth/node bound.')
    require(value is None or type(value) in (dict, list, bool, str, int, float), 'Unsupported JSON value.')
    if type(value) is int:
        natural(abs(value), 'JSON integer')
    elif type(value) is float:
        require(math.isfinite(value), 'JSON must be finite.')
    elif type(value) is str:
        require(len(value) <= 16384 and not any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value), 'Unsupported JSON text.')
    elif type(value) is list:
        for child in value:
            _json(child, depth+1, budget)
    elif type(value) is dict:
        for key, child in value.items():
            require(type(key) is str, 'JSON object keys must be strings.')
            _json(key, depth+1, budget); _json(child, depth+1, budget)


def canonical(value):
    _json(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(raw, maximum=JSON_LIMIT):
    require(type(raw) is bytes and 0 < len(raw) <= maximum, 'JSON exceeds its byte bound.')
    def pairs(items):
        value = {}
        for key, child in items:
            require(key not in value, 'Duplicate JSON key.')
            value[key] = child
        return value
    try:
        result = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(EvidenceError('JSON must be finite.')))
    except (ValueError, UnicodeError, RecursionError):
        raise EvidenceError('Expected strict bounded UTF-8 JSON.') from None
    _json(result)
    return result


def reference(value, kind, catalog_id):
    keys = {'kind', 'catalog_id', 'id', 'sha256'} | ({'revision'} if kind in ('model', 'task', 'profile') else set())
    closed(value, keys, kind + ' reference')
    require(value['kind'] == kind and value['catalog_id'] == catalog_id, 'Reference owner differs.')
    sha(value['sha256'])
    if kind == 'recipe_material':
        require(value['id'] == value['sha256'], 'Composition content identity differs.')
    elif kind == 'model':
        identifier(value['id'])
    else:
        require(type(value['id']) is str and 0 < len(value['id']) <= 64, 'Definition identifier differs.')
    if 'revision' in keys:
        natural(value['revision'], 'Definition revision', 1)
    return value


@boundary
def load_context(raw):
    """Read the existing canonical host projection, not an alternate policy schema.

    This proves bounded common identities. The host still validates its complete
    policy/source/grant/custody contract before execution and after publication.
    """
    value = decode(raw, CONTEXT_LIMIT)
    closed(value, ('format', 'static', 'composition', 'run_id', 'request_sha256'), 'Context')
    require(value['format'] == CONTEXT_FORMAT, 'Unsupported evidence context.', 'unsupported')
    require(canonical(value) == raw, 'Host context must retain its exact canonical bytes.')
    identifier(value['run_id']); sha(value['request_sha256'])
    static = value['static']
    closed(static, ('format', 'catalog_id', 'model', 'task', 'profile', 'probe', 'policy', 'inputs', 'parameters', 'execution_material_sha256'), 'Static context')
    require(static['format'] == 'extension_evidence_static_v1', 'Unsupported static evidence context.', 'unsupported')
    catalog_id = identifier(static['catalog_id'])
    reference(value['composition'], 'recipe_material', catalog_id)
    reference(static['model'], 'model', catalog_id); reference(static['task'], 'task', catalog_id)
    sha(static['execution_material_sha256'])
    profile, probe, policy = static['profile'], static['probe'], static['policy']
    require(type(policy) is dict and type(profile) is dict and type(probe) is dict, 'Missing admitted policy/profile/probe.')
    require(profile.get('format') == 'resolved_profile_v1' and probe.get('format') == 'probe_v1', 'Unsupported resolved profile/probe.', 'unsupported')
    reference(profile['definition'], 'profile', catalog_id)
    require(profile['task_sha256'] == static['task']['sha256'] == probe['task_sha256'] and profile['model_sha256'] == static['model']['sha256'], 'Task/model/probe identities differ.')
    require(profile['definition']['sha256'] == digest(policy['profile']) and static['task']['sha256'] == digest(policy['task']['definition']), 'Admitted definition identities differ.')
    require(probe['split'] == 'validation' and type(probe['members']) is list and probe['members'], 'Only explicit validation probes are supported.', 'unsupported')
    sha(probe['data_sha256']); sha(probe['selection_policy_sha256'])
    require(type(profile['capture']) is dict and type(profile['limits']) is dict, 'Capture and output bounds are required.')
    for key in ('manifest_bytes', 'shard_bytes', 'total_output_bytes'):
        natural(profile['limits'][key], key, 1, 512 * 1024**2)
    capture = profile['capture']
    require(capture['clock'] in ('optimizer_update', 'epoch', 'fit_pass'), 'Unsupported checkpoint clock.', 'unsupported')
    natural(capture['interval'], 'Capture interval', 1); natural(capture['max_snapshots'], 'Snapshot count', 1, 256)
    return value


def clocks(value, applicable):
    closed(value, ('optimizer_update', 'epoch', 'fit_pass', 'physical_time', 'wall_elapsed_ms'), 'Clocks')
    natural(value['wall_elapsed_ms'], 'Wall elapsed milliseconds')
    require(value['physical_time'] is None, 'Standalone evidence has no physical timeline.', 'unsupported')
    for key in ('optimizer_update', 'epoch', 'fit_pass'):
        if value[key] is not None:
            natural(value[key], key, 0 if key == 'optimizer_update' else 1)
    require(value.get(applicable) is not None, 'Applicable checkpoint clock is missing.')
    return value[applicable]


@boundary
def progress(stage, completed, total, metrics=None):
    """Construct the existing bounded stdout progress protocol; infer no progress."""
    require(type(stage) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', stage), 'Invalid progress stage.')
    natural(total, 'Progress total', 0, 1000000); natural(completed, 'Progress completed', 0, total)
    metrics = {} if metrics is None else metrics
    require(type(metrics) is dict and len(metrics) <= 8, 'Progress supports at most eight metrics.')
    for key, number in metrics.items():
        require(type(key) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', key) and type(number) in (int, float) and math.isfinite(number), 'Invalid progress metric.')
    return canonical(dict(format='tuldok_progress_v1', stage=stage, completed=completed, total=total, metrics=metrics)).decode()


def read_regular(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        before = os.fstat(source.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, 'Expected a bounded single-link regular file.', 'unavailable')
        raw = source.read(maximum+1); after = os.fstat(source.fileno())
    identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    require(identity(before) == identity(after) and len(raw) == after.st_size and len(raw) <= maximum, 'File changed during its bounded read.', 'unavailable')
    return raw


class Evidence:
    """Used A/B D2 writer; numerical values and produced array files are caller-owned."""
    @boundary
    def __init__(self, output, context_raw):
        self.context = load_context(context_raw); self.static = self.context['static']; self.policy = self.static['policy']
        self.output = Path(output); self.entries = []; self.observations = []; self.field_groups = []; self._inventory = {}; self._finished = False
        require(self.output.is_dir() and not self.output.is_symlink(), 'Select an existing output directory.', 'unavailable')
        self.write_raw('execution-context.json', context_raw)
        for filename, value in (('model-definition.json', self.policy['model_specification']), ('profile.json', self.static['profile']),
                ('output-policy.json', self.policy['output_policy']), ('probe.json', self.static['probe']), ('metric-definitions.json', self.policy['metric_definitions'])):
            self.write(filename, value)

    @boundary
    def write_raw(self, filename, raw):
        name(filename); require(type(raw) is bytes, 'Evidence bytes are unavailable.')
        maximum = self.static['profile']['limits']['manifest_bytes' if filename == 'model-evidence.json' else 'shard_bytes']
        require(len(raw) <= maximum, 'Evidence file exceeds its admitted bound.')
        fd = os.open(self.output / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
        self.blob(filename)

    def write(self, filename, value):
        self.write_raw(filename, canonical(value) + b'\n')

    @boundary
    def blob(self, filename):
        name(filename)
        raw = read_regular(self.output / filename, self.static['profile']['limits']['shard_bytes'])
        require(len(raw) > 0, 'Evidence blobs must not be empty.')
        value = dict(name=filename, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), media_type='application/json' if filename.endswith('.json') else 'application/octet-stream')
        require(filename not in self._inventory or self._inventory[filename] == value, 'An observed evidence file changed.')
        self._inventory[filename] = value
        require(sum(item['bytes'] for item in self._inventory.values()) <= self.static['profile']['limits']['total_output_bytes'], 'Evidence exceeds its admitted aggregate bound.')
        return copy.deepcopy(value)

    @boundary
    def transform(self, state_name):
        self.write('fitted-transform.json', dict(format='fitted_transform_v1', fit_scope='train', fit_members=self.policy['fit_members'], state=self.blob(state_name)))

    @boundary
    def checkpoint(self, point, observed_clocks, weights_name, metrics):
        require(not self._finished, 'Evidence is already finished.')
        capture = self.static['profile']['capture']; natural(point, 'Checkpoint point', 1)
        require(point == (len(self.entries)+1)*capture['interval'] and len(self.entries) < capture['max_snapshots'], 'Checkpoint cadence differs from admitted capture.')
        require(clocks(observed_clocks, capture['clock']) == point and (not self.entries or observed_clocks['wall_elapsed_ms'] >= self.entries[-1]['clocks']['wall_elapsed_ms']), 'Checkpoint clocks differ or move backwards.')
        require(type(metrics) is dict and set(metrics) == {item['id'] for item in self.policy['metric_definitions']}, 'Checkpoint metric membership differs.')
        run_id = self.context['run_id']; transform = self.blob('fitted-transform.json')
        descriptor = dict(format='checkpoint_state_v1', run_id=run_id, model_sha256=self.static['model']['sha256'], weights=self.blob(weights_name), fitted_transform_sha256=transform['sha256'], clocks=observed_clocks)
        filename = f'checkpoint-{point}.json'; self.write(filename, descriptor); ref = self.blob(filename)
        self.entries.append(dict(checkpoint=ref, model_sha256=descriptor['model_sha256'], fitted_transform_sha256=transform['sha256'], clocks=copy.deepcopy(observed_clocks), selected=False, state='retained'))
        probe = self.blob('probe.json')
        for metric in self.policy['metric_definitions']:
            value = metrics[metric['id']]
            require(type(value) in (int, float) and math.isfinite(value) and metric['domain']['minimum'] <= value <= metric['domain']['maximum'], 'Observation is outside its declared metric domain.')
            self.observations.append(dict(run_id=run_id, checkpoint_sha256=ref['sha256'], probe_sha256=probe['sha256'], metric_id=metric['id'], split='validation', clocks=copy.deepcopy(observed_clocks), state='observed', value=value, sample_count=len(self.static['probe']['members'])))
        bindings = {(item['field_id'], item['member_id']): item for item in self.policy['field_outputs'] if item['clock'] == point}
        for field in self.policy['fields']:
            cases = []
            for member in self.static['probe']['members']:
                binding = bindings[field['id'], member['frozen_reference']['member_id']]
                cases.append(dict(member_ordinal=member['ordinal'], predicted=self.blob(binding['prediction']), reference=self.blob(binding['reference']), mask=None,
                    geometry_sha256=digest(field['geometry']), time_units='not_applicable', gauge_policy_sha256=field['contract']['output']['gauge_policy_sha256']))
            self.field_groups.append(dict(run_id=run_id, checkpoint_sha256=ref['sha256'], probe_sha256=probe['sha256'], cases=cases))

    @boundary
    def finish(self):
        require(not self._finished and len(self.entries) == self.static['profile']['capture']['max_snapshots'], 'Required checkpoint cadence is incomplete or already finished.')
        for member in self.policy['probe_inputs']:
            for field in member['fields']:
                self.blob(field['artifact'])
        self.entries[-1]['selected'] = True
        self.write('checkpoints.json', dict(format='checkpoint_index_v1', run_id=self.context['run_id'], entries=self.entries))
        self.write('observations.json', self.observations); self.write('field-index.json', self.field_groups)
        self.write('model-evidence.json', dict(format='model_evidence_v1', schema_version=1,
            producer=dict(catalog_id=self.static['catalog_id'], run_id=self.context['run_id'], material_sha256=self.context['composition']['sha256'], adapter_sha256=self.policy['adapter_sha256']),
            task=self.static['task'], model_sha256=self.static['model']['sha256'], data_sha256=self.static['probe']['data_sha256'], profile_sha256=digest(self.static['profile']),
            probe=self.blob('probe.json'), metric_definitions=self.blob('metric-definitions.json'), checkpoints=self.blob('checkpoints.json'), observations=[self.blob('observations.json')], field_index=self.blob('field-index.json'),
            coverage=dict(required=self.static['profile']['required_capabilities'], optional=[], omitted=[]),
            inference=dict(mode='eval_no_grad', fitted_transforms=[self.blob('fitted-transform.json')], output_policy_sha256=digest(self.policy['output_policy']))))
        for filename in tuple(self._inventory):
            self.blob(filename)
        self._finished = True
        return self.inventory()

    def inventory(self):
        """Blobs actually consumed/written by this builder, not an export permission."""
        return [copy.deepcopy(self._inventory[key]) for key in sorted(self._inventory)]


@boundary
def validate_projection(value, context_raw, blobs):
    """Bind a current public verified projection to downloaded whole-hash bytes.

    Opaque model/array bytes remain opaque. The host owns complete schema/policy
    verification; this helper checks the exact common D2 identity/Blob closure.
    """
    context = load_context(context_raw); static = context['static']; policy = static['policy']
    _json(value)
    require(type(value) is dict and value.get('state') == 'verified' and not any(key.startswith('_') for key in value), 'Expected a public verified evidence projection.')
    require(type(blobs) is dict and all(type(key) is str and type(raw) is bytes for key, raw in blobs.items()), 'Downloaded bytes are required.')
    def document(filename):
        require(filename in blobs, 'Required evidence document is unavailable.', 'unavailable')
        return decode(blobs[filename])
    def blob(ref):
        closed(ref, ('name', 'bytes', 'sha256', 'media_type'), 'Blob')
        name(ref['name']); sha(ref['sha256']); natural(ref['bytes'], 'Blob bytes', 1, JSON_LIMIT)
        raw = blobs.get(ref['name'])
        require(type(raw) is bytes and len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Downloaded Blob differs from evidence identity.')
        return raw
    require(blobs.get('execution-context.json') == context_raw, 'Downloaded context differs.')
    manifest = value['manifest']; require(canonical(manifest) == canonical(document('model-evidence.json')), 'Manifest projection differs from retained bytes.')
    closed(manifest, ('format', 'schema_version', 'producer', 'task', 'model_sha256', 'data_sha256', 'profile_sha256', 'probe', 'metric_definitions', 'checkpoints', 'observations', 'field_index', 'coverage', 'inference'), 'Evidence manifest')
    require(manifest['format'] == 'model_evidence_v1' and type(manifest['schema_version']) is int and manifest['schema_version'] == 1, 'Unsupported evidence manifest.', 'unsupported')
    require(canonical(manifest['producer']) == canonical(dict(catalog_id=static['catalog_id'], run_id=context['run_id'], material_sha256=context['composition']['sha256'], adapter_sha256=policy['adapter_sha256']))
        and canonical(manifest['task']) == canonical(static['task']) and manifest['model_sha256'] == static['model']['sha256']
        and manifest['data_sha256'] == static['probe']['data_sha256'] and manifest['profile_sha256'] == digest(static['profile']), 'Manifest admitted identities differ.')
    for key, expected in (('probe', static['probe']), ('metric_definitions', policy['metric_definitions'])):
        require(canonical(decode(blob(manifest[key]))) == canonical(expected) == canonical(value[key]), 'Public ' + key + ' differs.')
    require(value['metric_definition_hashes'] == {item['id']: digest(item) for item in policy['metric_definitions']}, 'Metric-definition identities differ.')
    require(canonical(value['model_definition']) == canonical(document('model-definition.json')) == canonical(policy['model_specification']), 'Model definition differs.')
    require(canonical(document('profile.json')) == canonical(static['profile']) and canonical(document('output-policy.json')) == canonical(policy['output_policy']), 'Retained profile/output policy differs.')
    transform = document('fitted-transform.json')
    closed(transform, ('format', 'fit_scope', 'fit_members', 'state'), 'Fitted transform')
    require(transform['format'] == 'fitted_transform_v1' and transform['fit_scope'] == 'train' and canonical(transform['fit_members']) == canonical(policy['fit_members']), 'Fitted member lineage differs.')
    blob(transform['state'])
    inference = manifest['inference']; closed(inference, ('mode', 'fitted_transforms', 'output_policy_sha256'), 'Inference binding')
    require(inference['mode'] == 'eval_no_grad' and inference['output_policy_sha256'] == digest(policy['output_policy']) and len(inference['fitted_transforms']) == 1 and blob(inference['fitted_transforms'][0]) == blobs['fitted-transform.json'], 'Inference transform/output binding differs.')
    index = decode(blob(manifest['checkpoints']))
    require(canonical(index) == canonical(value['checkpoint_index']) and index['format'] == 'checkpoint_index_v1' and index['run_id'] == context['run_id'], 'Checkpoint index differs.')
    require(len(index['entries']) == static['profile']['capture']['max_snapshots'] and sum(item['selected'] is True for item in index['entries']) == 1, 'Checkpoint coverage differs.')
    descriptors = []
    for ordinal, entry in enumerate(index['entries'], 1):
        require(clocks(entry['clocks'], static['profile']['capture']['clock']) == ordinal*static['profile']['capture']['interval'], 'Checkpoint clock differs.')
        descriptor = decode(blob(entry['checkpoint'])); descriptors.append(descriptor); blob(descriptor['weights'])
        require(descriptor['run_id'] == context['run_id'] and descriptor['model_sha256'] == entry['model_sha256'] == static['model']['sha256']
            and descriptor['fitted_transform_sha256'] == entry['fitted_transform_sha256'] == hashlib.sha256(blobs['fitted-transform.json']).hexdigest()
            and canonical(descriptor['clocks']) == canonical(entry['clocks']), 'Checkpoint/state binding differs.')
    require(canonical(descriptors) == canonical(value['checkpoints']), 'Public checkpoint descriptors differ.')
    require(len(manifest['observations']) == 1 and canonical(decode(blob(manifest['observations'][0]))) == canonical(value['observations']), 'Observation projection differs.')
    for group in decode(blob(manifest['field_index'])):
        for case in group['cases']:
            blob(case['predicted']); blob(case['reference'])
    return dict(state='passed', scope='common_d2_identity_and_whole_blob_closure', numerical='not_evaluated')
