"""Verify actual native full-state custody without opening a project or model.

The known pre-import header is reconstructed in memory by removing only the
acknowledged additions. No duplicate manifest, payload or scientific fixture is
created. Existing payload bodies other than the eight new blobs are not rehashed.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import stat
import struct
import time


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
JOB = RUN / 'native-full-state-import-01'
BUNDLE = RUN / 'native-full-state-comparisons-02'
REPORT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/native-full-state-ingestion-custody-01.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def checked(path, expected):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('Native custody member must be an independent ordinary file')
    raw = path.read_bytes()
    after = path.stat()
    if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or len(raw) != expected['bytes'] or sha(raw) != expected['sha256']):
        raise ValueError('Retained bytes or file identity changed during audit')
    return raw


def run():
    if REPORT.exists():
        raise ValueError('Fresh audit report required; preserve original evidence')
    began = time.monotonic()
    status_raw = (JOB / 'status.json').read_bytes()
    status = json.loads(status_raw)
    if (status['status'] != 'seven_full_state_packets_native_import_and_metadata_passed'
            or len(status['imports']) != 7 or len(status['metadata_checks']) != 7
            or any(r.get('status') != 200 for r in status['requests'])):
        raise ValueError('Original native sequence has not completely acknowledged')
    header = PROJECT / '.tuldok/project.json'
    header_raw = header.read_bytes()
    manifest = json.loads(header_raw)
    if (manifest['checkpoint'] != status['last_acknowledged_checkpoint']
            or manifest['project_id'] != status['project_id'] or encode(manifest) != header_raw
            or sha(encode({k: v for k, v in manifest.items() if k != 'checkpoint'})) != manifest['checkpoint']):
        raise ValueError('Actual canonical header differs from its acknowledged checkpoint')
    bundle_raw = (BUNDLE / 'bundle.json').read_bytes()
    source_raw = (BUNDLE / 'source-project.json').read_bytes()
    bundle, source = json.loads(bundle_raw), json.loads(source_raw)
    if sha(bundle_raw) != status['bundle_sha256'] or sha(source_raw) != status['manifest_sha256']:
        raise ValueError('Original imported source preparation changed')
    keys = []
    for call in status['requests']:
        if call['operation'] not in ('project_files.source_register', 'project_files.measurement_import'):
            continue
        if call['status'] != 200:
            raise ValueError('Mutation acknowledgement missing')
        request = read(JOB / call['request'])
        keys.append(request['arguments']['request_key'])
    if len(keys) != 8 or len(set(keys)) != 8:
        raise ValueError('Exactly eight independent native mutations required')
    additions = {('project_operation', status['project_id'] + '.' + sha(key.encode())) for key in keys}
    source_identity = status['project_id'] + '.' + status['manifest_sha256']
    additions.update({('project_snapshot', source_identity), ('project_artifact', status['manifest_sha256'])})
    for entry in status['imports']:
        additions.update({('project_artifact', entry['packet_sha256']),
                          ('measurement_source', entry['origin_catalog_id'] + '.' + entry['packet_sha256'])})
    if len(additions) != 24:
        raise ValueError('Unexpected duplicate native additions')
    indexed = {(r['family'], r['id']): r for r in manifest['inventory']}
    roots = {(r['family'], r['id']) for r in manifest['roots']}
    if len(indexed) != len(manifest['inventory']) or roots != set(indexed) or not additions <= roots:
        raise ValueError('Canonical inventory/root/addition identities differ')
    previous = {k: v for k, v in manifest.items() if k != 'checkpoint'}
    previous['inventory'] = [r for r in manifest['inventory'] if (r['family'], r['id']) not in additions]
    previous['roots'] = [r for r in manifest['roots'] if (r['family'], r['id']) not in additions]
    previous['receipts'] = [r for r in manifest['receipts'] if r['request_key'] not in keys]
    previous['checkpoint'] = sha(encode(previous))
    original_preparation = read(RUN / 'native-method-review-revisions-06/preparation.json')
    if (previous['checkpoint'] != original_preparation['source_checkpoint']
            or sha(encode(previous)) != original_preparation['source_manifest_sha256']
            or len(previous['inventory']) != 53874 or len(previous['receipts']) != 3):
        raise ValueError('Original full inventory/roots/receipts do not reconstruct exactly')

    declarations, measurements, source_document, operations = {}, {}, None, []
    canonical_bytes, family_counts = 0, Counter()
    for entry in manifest['inventory']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Canonical member path escapes metadata root')
        raw = checked(PROJECT / '.tuldok' / relative, entry)
        canonical_bytes += len(raw)
        family_counts[entry['family']] += 1
        identity = entry['family'], entry['id']
        if entry['family'] == 'project_artifact':
            declarations[entry['id']] = json.loads(raw)
        elif entry['family'] == 'measurement_source':
            measurements[entry['id']] = json.loads(raw)
        elif identity == ('project_snapshot', source_identity):
            source_document = json.loads(raw)
        elif identity in additions and entry['family'] == 'project_operation':
            document = json.loads(raw)
            if document['request_key'] not in keys:
                raise ValueError('New operation receipt is not one of the acknowledged requests')
            operations.append(document)
    if len(operations) != 8 or source_document is None:
        raise ValueError('Registered source or mutation receipts missing')
    if (source_document['origin_catalog_id'] != status['project_id']
            or source_document['value']['id'] != source['id']
            or source_document['value']['manifest'] != source):
        raise ValueError('Actual registered native source differs from original manifest')
    expected_sources = {tuple(r) for r in status['preexisting_source_identities']}
    expected_sources.update((r['origin_catalog_id'], r['packet_sha256']) for r in status['imports'])
    if {(d['origin_catalog_id'], d['id']) for d in measurements.values()} != expected_sources or len(measurements) != 16:
        raise ValueError('Prior or new measurement-source population differs')
    manifest_blob = dict(sha256=status['manifest_sha256'], bytes=len(source_raw))
    if declarations.get(manifest_blob['sha256']) != dict(format='project_artifact_v1', **manifest_blob):
        raise ValueError('Registered source manifest artifact declaration differs')
    checked(PROJECT / 'artifacts/sha256' / manifest_blob['sha256'], manifest_blob)

    measured, unavailable, observations = 0, 0, 0
    scalar_bits = hashlib.sha256()
    packets = []
    subjects = set()
    for entry, admitted in zip(bundle['packets'], status['imports'], strict=True):
        if entry['sha256'] != admitted['packet_sha256'] or entry['path'] != admitted['path']:
            raise ValueError('Original packet order/identity differs')
        blob = {k: entry[k] for k in ('sha256', 'bytes')}
        if declarations.get(blob['sha256']) != dict(format='project_artifact_v1', **blob):
            raise ValueError('Actual packet artifact declaration differs')
        document = measurements[admitted['origin_catalog_id'] + '.' + entry['sha256']]
        reference = next(r for r in source['data'] if r['sha256'] == entry['sha256'])
        expected_source = dict(origin_catalog_id=status['project_id'], project_id=source['id'],
            manifest_sha256=status['manifest_sha256'], reference_id=reference['id'],
            path=reference['path'], measured_sha256=entry['sha256'])
        if document['source'] != expected_source or document['blob'] != blob:
            raise ValueError('Actual measurement source is not bound to the exact original packet')
        raw = checked(PROJECT / 'artifacts/sha256' / entry['sha256'], blob)
        packet = json.loads(raw)
        counts = Counter(r['state'] for r in packet['observations'])
        if (counts != Counter(measured=entry['measured'], unavailable=entry['unavailable'])
                or len(packet['observations']) != admitted['observation_count']
                or len(packet['subjects']) != admitted['subject_count']
                or len(packet['cases']) != admitted['case_count']):
            raise ValueError('Native retained scalar/state/case/subject counts differ')
        for row in packet['observations']:
            if row['state'] != 'measured':
                continue
            value = row['value']
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Retained measured scalar is not a finite number')
            scalar_bits.update(bytes.fromhex(entry['sha256']))
            scalar_bits.update(row['id'].encode())
            scalar_bits.update(struct.pack('!d', value))
        measured += counts['measured']
        unavailable += counts['unavailable']
        observations += len(packet['observations'])
        subjects.update(r['id'] for r in packet['subjects'])
        packets.append(dict(path=entry['path'], native_blob_sha256=entry['sha256'], bytes=len(raw),
            subjects=len(packet['subjects']), cases=len(packet['cases']), observations=len(packet['observations']),
            measured=counts['measured'], unavailable=counts['unavailable'], original_capture_bytes_exact=True))
    if (measured, unavailable, observations, len(subjects)) != (129626, 46954, 176580, 12):
        raise ValueError('Complete native aggregate differs from the qualified source population')
    if header.read_bytes() != header_raw or (JOB / 'status.json').read_bytes() != status_raw:
        raise ValueError('Native selection/import evidence changed during the audit')
    result = dict(status='all_current_canonical_bytes_and_native_full_state_packet_custody_exact',
        created_unix=time.time(), producer=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__).read_bytes())),
        project_id=manifest['project_id'], checkpoint=manifest['checkpoint'], canonical_header_sha256=sha(header_raw),
        canonical_records=len(indexed), canonical_record_bytes=canonical_bytes, canonical_families=dict(family_counts),
        reconstructed_previous_checkpoint=previous['checkpoint'], reconstructed_previous_header_sha256=sha(encode(previous)),
        all53874previous_record_inventory_entries_roots_and_receipts_exact=True, added_records=24,
        previous_measurement_sources_preserved=9, current_measurement_sources=16,
        new_packet_blobs_freshly_hashed=7, new_manifest_blob_freshly_hashed=True, packets=packets,
        measured_scalars=measured, explicit_unavailable=unavailable, observations=observations,
        execution_subjects=len(subjects), measured_binary64_identity_sha256=scalar_bits.hexdigest(),
        raw_original_capture_bytes_preserve_all_scalar_precision=True,
        previous_payload_bodies_rehashed=False, previous_payload_declarations_preserved=True,
        new_neural_forwards=0, SQL_opened=False, payload_copies=0, native_mutations=0,
        native_UI_or_chart_queries_qualified=False, final_portable_cold_qualified=False,
        complete_portfolio_benchmark=False, clean_reproduction_qualified=False,
        elapsed_seconds=time.monotonic()-began,
        scope='Fresh actual hashes of all current canonical files and all eight new blobs; exact pre-import manifest reconstruction against independently retained hash. New payload bytes equal original qualified captures. Original9sources and all53,874old record references retained. Old scientific payload bodies, actual chart queries, all-model timing/retraining and final-source cold lifecycle remain separate requirements.')
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'checkpoint', 'canonical_records', 'canonical_record_bytes', 'measured_scalars', 'explicit_unavailable', 'elapsed_seconds')}))


if __name__ == '__main__':
    run()
