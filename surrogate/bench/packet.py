"""Turn a benchmark result document into the packets Tuldok imports.

Tuldok's `retained_measurements_v1` packet is a complete subject x case x metric
table whose every value points at the same number inside an embedded source
document. Here that source is the result document itself, cut down to the
packet's subjects. Packets from one result share scope, cases and metrics and
never share a subject, which is what Tuldok requires to compare them together.

The limits below are Tuldok's (retained_measurements.py); the contract test runs
its validator over what this module writes.
"""
import base64
import hashlib
import json

from . import results

PACKET_FORMAT = 'retained_measurements_v1'
MAX_ROWS = 100_000
MAX_DIMENSIONS = 32
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_PACKET_BYTES = 32 * 1024 * 1024
SOURCE_ID = 'result'
_TYPES = {bool: 'boolean', int: 'integer', float: 'number', str: 'string'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _definition(pointer, value):
    return {'source_id': SOURCE_ID, 'pointer': pointer, 'sha256': hashlib.sha256(_canonical(value)).hexdigest()}


def _escape(token):
    return token.replace('~', '~0').replace('/', '~1')


def _dimensions(entries, scope):
    """Declare one table column per attribute that entries of this kind carry."""
    kinds = {}
    for entry in entries:
        for name, value in entry.items():
            if name in ('id', 'label') or value is None:
                continue
            if type(value) not in _TYPES:
                raise ValueError(f'{scope} attribute {name!r} must be a string, number or boolean.')
            kinds.setdefault(name, set()).add(_TYPES[type(value)])
    columns = []
    for name, found in sorted(kinds.items()):
        if found == {'integer', 'number'}:
            found = {'number'}
        if len(found) != 1:
            raise ValueError(f'{scope} attribute {name!r} mixes {sorted(found)} values.')
        columns.append({'id': name, 'label': name.replace('_', ' '), 'type': found.pop(), 'units': None,
                        'scope': scope})
    return columns


def _values(entry, dimensions):
    """An entry's column values; a number column always holds floats."""
    row = {}
    for dimension in dimensions:
        value = entry.get(dimension['id'])
        row[dimension['id']] = float(value) if dimension['type'] == 'number' and value is not None else value
    return row


def _packet(document, subjects, part, parts):
    """One packet holding `subjects`, with its own embedded copy of their results."""
    names = {subject['id'] for subject in subjects}
    source = {**document, 'subjects': subjects,
              'results': [result for result in document['results'] if result['subject'] in names]}
    source_bytes = _canonical(source)
    suite = source['suite']
    title = suite['label'] if parts == 1 else f"{suite['label']} · part {part} of {parts}"
    subject_dimensions = _dimensions(document['subjects'], 'subject')
    case_dimensions = _dimensions(document['cases'], 'case')
    if len(subject_dimensions) + len(case_dimensions) > MAX_DIMENSIONS:
        raise ValueError(f'Subjects and cases together carry more than {MAX_DIMENSIONS} attributes.')
    observations = []
    for index, result in enumerate(source['results']):
        for metric in source['metrics']:
            value = result['values'][metric['id']]
            measured = value is not None
            if measured:
                state, reason = 'measured', None
            elif result['failure']:
                # Tuldok accepts one line of at most 1000 characters.
                state, reason = 'failed', ' '.join(result['failure'].split())[:1000]
            else:
                state, reason = 'unsupported', 'This metric does not apply to this subject and case.'
            observations.append({
                'id': f"r{index}.{metric['id']}",
                'subject_id': result['subject'], 'case_id': result['case'], 'metric_id': metric['id'],
                'state': state, 'value': value, 'reason': reason, 'dimensions': {},
                'reference': {'source_id': SOURCE_ID,
                              'pointer': f"/results/{index}/values/{_escape(metric['id'])}" if measured else None}})
    packet = {
        'format': PACKET_FORMAT, 'schema_version': 1, 'title': title,
        'scope': {'id': suite['id'], 'label': suite['label'], 'population': suite['population'],
                  'definition': _definition('/suite', suite)},
        'dimensions': subject_dimensions + case_dimensions,
        'subjects': [{'id': subject['id'], 'label': subject['label'],
                      'dimensions': _values(subject, subject_dimensions),
                      'reference': {'source_id': SOURCE_ID, 'pointer': f'/subjects/{index}'}}
                     for index, subject in enumerate(subjects)],
        'cases': [{'id': case['id'], 'dimensions': _values(case, case_dimensions),
                   'reference': {'source_id': SOURCE_ID, 'pointer': f'/cases/{index}'}}
                  for index, case in enumerate(source['cases'])],
        'metrics': [{'id': metric['id'], 'label': metric['label'], 'units': metric['units'],
                     'definition': _definition(f'/metrics/{index}', metric)}
                    for index, metric in enumerate(source['metrics'])],
        'observations': observations,
        'sources': [{'id': SOURCE_ID, 'label': 'Benchmark result document', 'kind': 'json',
                     'bytes': len(source_bytes), 'sha256': hashlib.sha256(source_bytes).hexdigest(),
                     'data_base64': base64.b64encode(source_bytes).decode('ascii')}]}
    return len(source_bytes), json.dumps(packet, separators=(',', ':'), allow_nan=False).encode('utf-8')


def build(document):
    """Return the packets (bytes) for a result document, as few as Tuldok's limits allow."""
    results.validate(document)
    subjects = document['subjects']
    rows_per_subject = len(document['cases']) * len(document['metrics'])
    per_packet = min(len(subjects), MAX_ROWS // rows_per_subject)
    if per_packet < 1:
        raise ValueError('One subject alone exceeds the packet row limit; split the suite into smaller ones.')
    while True:
        groups = [subjects[start:start + per_packet] for start in range(0, len(subjects), per_packet)]
        built = [_packet(document, group, index + 1, len(groups)) for index, group in enumerate(groups)]
        if all(source <= MAX_SOURCE_BYTES and len(packet) <= MAX_PACKET_BYTES for source, packet in built):
            return [packet for _, packet in built]
        if per_packet == 1:
            raise ValueError('One subject alone exceeds the packet size limit; split the suite into smaller ones.')
        per_packet = (per_packet + 1) // 2
