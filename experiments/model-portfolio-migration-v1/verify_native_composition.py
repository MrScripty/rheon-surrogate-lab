"""Literal preservation audit of the real native three-source publication.

Original portable manifests, record bytes and payload bytes are the oracle.
This does not import native composition/decoder code, SQL or model runtimes.
It proves custody, not cold lifecycle, UI usability or scientific equivalence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
import time

from verify_native_portfolio_fork import digest, inventory

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/model-portfolio-migration-v1'
# These identities come from the independent, completed source preservation
# receipts, before the native composition implementation ran.
INPUTS = (
    (RUN / 'native-composition-primary-01/root/project',
     '0775213f6a11d93f9d80db80ea76daa6e80e2fcd0194a673dc2e8d093d644570',
     'a1f3abc86bbbf8c3dcd5f8f7426c3193ba0c0ff07ab7fb80f8b94b8984914001'),
    (RUN / 'full-current-history-cold-01/root/project',
     '5f9281c0595bc0d09c81d9291ef40e86f3cd82cafa8957fb009cd89c1982e7ad',
     '720a9c4472fb92e934d9a0fb30f1292b40958b89ad34dde1b5b27dfd1127a347'),
    (Path('/tmp/rheon-archive-owner-1eju7eaw/projects/rheon-archived-history-01'),
     '6a547cd27a2c6f05d2760a31618de9855c99acb741768468754b192b42a42860',
     '6cfad7057fa10a9aa7281d032fc1f3e5f552470d2a4893671a0e8878ac000b2d'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    # The persisted format explicitly hashes UTF-8, sorted, indented JSON.
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def document(records, key):
    require(key in records, 'Required canonical record is absent: ' + str(key))
    return json.loads(records[key][1].read_bytes())


def independent(original, retained):
    a, b = original.stat(), retained.stat()
    require((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino),
            'Publication shares a source file inode')
    require(stat.S_IMODE(a.st_mode) == stat.S_IMODE(b.st_mode),
            'Publication changed an original file mode')


def verify(inputs, destination, request, report):
    require(not report.exists(), 'Fresh composition verification report required')
    started = time.monotonic()
    final_raw = (destination / '.tuldok/project.json').read_bytes()
    final, retained = inventory(destination)
    identity = final['project_id']
    provenance = document(retained, ('project_composition', identity))
    overlay = document(retained, ('project_fork', identity))
    arguments = request['arguments']
    for value in (provenance, overlay):
        require(value['destination_project_id'] == identity
                and all(value[key] == arguments[key] for key in ('directory', 'name', 'request_key')),
                'Publication provenance differs from the retained logical request')
    require(len(provenance['sources']) == len(inputs), 'Composition source population differs')
    require(len({s['project_id'] for s in provenance['sources']}) == len(inputs)
            and identity not in {s['project_id'] for s in provenance['sources']},
            'Publication is not independent of all source projects')
    original_keys, alternate_keys, copied, declarations = set(), set(), {}, set()
    observations = memberships = notes = 0
    source_results = []
    source_manifests = []
    primary_records = None
    for number, ((source, checkpoint, manifest_sha), membership) in enumerate(zip(inputs, provenance['sources'])):
        raw = (source / '.tuldok/project.json').read_bytes()
        require(sha(raw) == manifest_sha, 'Original source manifest differs from its preservation receipt')
        descriptor, originals = inventory(source)
        require(descriptor['checkpoint'] == checkpoint
                and all(membership[k] == descriptor[k] for k in ('project_id', 'name', 'checkpoint')),
                'Source project/checkpoint differs from native provenance')
        source_manifests.append((source, raw))
        expected_membership = {}
        alternate = membership['run_observations']
        require(set(alternate) <= {i for f, i in originals if f == 'run_history'},
                'Alternate proof mapping refers to an absent original Run')
        for key, (entry, path) in originals.items():
            family, record_id = key
            original_keys.add(key)
            original = json.loads(path.read_bytes())
            envelope_sha = sha(canonical(dict(family=family, id=record_id, document=original)))
            expected_membership.setdefault(family, {})[record_id] = envelope_sha
            if family == 'run_history' and record_id in alternate:
                observed_key = ('run_observation', alternate[record_id])
                observed = document(retained, observed_key)
                base = document(retained, key)
                require(canonical(observed['observation']) == path.read_bytes()
                        and observed_key[1] == sha(path.read_bytes()) == observed['sha256'],
                        'Alternate Run proof does not preserve the complete original document')
                require(base != original
                        and {k: v for k, v in base.items() if k != 'admission_proof'}
                        == {k: v for k, v in original.items() if k != 'admission_proof'}
                        and base['admission_proof']['admission'] == original['admission_proof']['admission'],
                        'Alternate observation changed more than the permitted approval proof')
                alternate_keys.add(observed_key)
                observations += 1
            else:
                require(key in retained and retained[key][1].read_bytes() == path.read_bytes(),
                        'Original canonical fact changed or disappeared: ' + str(key))
                independent(path, retained[key][1])
            if family == 'project_artifact':
                declarations.add(record_id)
                before = source / 'artifacts/sha256' / record_id
                if before.exists():
                    require(stat.S_ISREG(before.lstat().st_mode)
                            and before.stat().st_size == original['bytes'] and digest(before) == record_id,
                            'Original source payload differs from its intrinsic declaration')
                    copied.setdefault(record_id, []).append((before, original['bytes']))
        require(membership['records'] == expected_membership, 'Source membership omits or changes an original envelope')
        memberships += len(originals)
        source_results.append(dict(path=str(source), project_id=descriptor['project_id'],
                                   checkpoint=checkpoint, manifest_sha256=manifest_sha,
                                   records=len(originals), alternate_Run_proofs=len(alternate)))
        if number == 0:
            primary_records = originals
    require(set(overlay['artifacts']['copied']) == set(copied)
            and set(overlay['artifacts']['missing_regenerable']) == declarations - set(copied),
            'Destination payload disposition differs from all original sources')
    actual_payloads = {p.name for p in (destination / 'artifacts/sha256').iterdir()}
    require(actual_payloads == set(copied), 'Published payload file membership differs')
    for intrinsic, original_files in copied.items():
        after = destination / 'artifacts/sha256' / intrinsic
        require(stat.S_ISREG(after.lstat().st_mode) and all(after.stat().st_size == size for _, size in original_files)
                and digest(after) == intrinsic, 'Published payload is missing or changed')
        for before, _ in original_files:
            independent(before, after)
    expected_keys = original_keys | alternate_keys | {('project_composition', identity), ('project_fork', identity)}
    destinations = {}
    for mapping in overlay['mapping']:
        family = 'workbench_record' if mapping['family'] == 'workbench_text' else mapping['family']
        key = family, mapping['destination_id']
        require(key not in destinations and key not in original_keys,
                'Editable destination mapping repeats or reuses an original identity')
        require((mapping['family'], mapping['source_id']) in primary_records,
                'Editable destination mapping does not belong to the primary source')
        destinations[key] = mapping
    expected_keys.update(destinations)
    require(set(retained) == expected_keys, 'Final canonical population has omissions or unexplained additions')
    for (family, record_id), (_, path) in primary_records.items():
        if family != 'project_file':
            continue
        old = json.loads(path.read_bytes())
        if old['project_id'] != provenance['sources'][0]['project_id']:
            continue
        matches = [(key, m) for key, m in destinations.items()
                   if m['family'] == family and m['source_id'] == record_id]
        require(len(matches) == 1, 'Primary authored file has no unique editable destination')
        key, mapping = matches[0]
        new = document(retained, key)
        require(new['project_id'] == identity and new['kind'] == old['kind']
                and new['path'] == old['path'] and new['history'] == old['history']
                and new['current'] == old['current']
                and mapping['source_revision'] == old['current']['revision'],
                'Editable authored file lost exact original content/revision history')
        if not new['current'].get('deleted', False):
            relative = Path(new['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'Authored path escapes project')
            before, after = inputs[0][0] / relative, destination / relative
            require(before.read_bytes() == after.read_bytes() == new['current']['text'].encode('utf-8'),
                    'Materialized authored text differs from its original captured content')
            independent(before, after)
        notes += old['kind'] == 'note'
    for source, raw in source_manifests:
        require((source / '.tuldok/project.json').read_bytes() == raw, 'Source changed during audit')
    require((destination / '.tuldok/project.json').read_bytes() == final_raw, 'Destination changed during audit')
    result = dict(schema='rheon.native-composition-preservation.v1', status='all_source_records_payloads_and_primary_authored_files_exact',
                  created_unix=time.time(), destination=str(destination), project_id=identity,
                  checkpoint=final['checkpoint'], manifest_sha256=sha(final_raw), sources=source_results,
                  original_memberships=memberships, retained_records=len(retained),
                  families=dict(Counter(f for f, i in retained)), alternate_Run_proofs=observations,
                  original_present_payloads_exact=len(copied), explicit_missing_dependencies=len(declarations-set(copied)),
                  editable_mappings=len(destinations), editable_notes_exact=notes,
                  independent_inodes=True, original_modes_exact=True, source_manifests_unchanged=True,
                  destination_manifest_unchanged=True, elapsed_seconds=time.monotonic()-started,
                  runner_sha256=digest(Path(__file__)), SQL_opened=False, models_executed=0,
                  limitations='Literal portable custody only; native cold lifecycle, real UI and benchmark/reproduction acceptance remain separate.')
    with report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT/'projects/rheon-model-portfolio-unified-04')
    parser.add_argument('--request', type=Path, default=RUN/'native-composition-authoring-01/compose-after-source-fix-01.request.json')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = verify(INPUTS, args.destination.resolve(), json.loads(args.request.read_bytes()), args.report.resolve())
    print(json.dumps({key: result[key] for key in ('status','original_memberships','retained_records','original_present_payloads_exact','editable_notes_exact')}))
