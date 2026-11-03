"""Bind retained expanded-variant evidence to current source availability.

This migration input reads JSON and hashes files; it imports no model or runtime.
Hash agreement establishes byte availability, not semantic adapter equivalence.
Retire this inventory tool after native project accounting owns these facts.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/plans/full-ai-data-diversity-v1'
CONFIG_KEYS = (
    'transport_pair', 'shared_pair', 'residual_pair', 'build', 'control_build',
    'sampler_build', 'fusion_mode', 'modes', 'horizons', 'case_selection',
    'bank', 'normalization', 'output_rate', 'output_rates', 'output_dt_s',
    'internal_substeps', 'groups', 'student_stages', 'scope', 'gpu_timing_scope',
)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def local_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Evidence path leaves the selected lab root')
    return path


def observe_claim(root, name, expected):
    """Observe the exact declared source; never select a nearby replacement."""
    try:
        path = local_path(root, name)
    except ValueError:
        return dict(path=name, expected_sha256=expected, status='outside_lab_scope')
    if not path.is_file():
        return dict(path=name, expected_sha256=expected, status='missing')
    actual = digest(path)
    return dict(path=name, expected_sha256=expected, current_sha256=actual,
                status='exact' if actual == expected else 'changed')


def protocol_index(directory):
    """Index actual bytes, so a filename similarity cannot bind a protocol."""
    index = defaultdict(list)
    for path in sorted(directory.rglob('*.json')):
        index[digest(path)].append(path)
    return index


def read_json(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Evidence document must be a JSON object: ' + str(path))
    return value, hashlib.sha256(raw).hexdigest()


def audit(root, worklist_path, protocols):
    worklist, worklist_sha = read_json(worklist_path)
    variants = [entry for entry in worklist['named_entries']
                if entry['item_id'].startswith('expanded-variants/')]
    index = protocol_index(protocols)
    documents = {}
    claims = {}

    def inspect(path):
        relative = str(path.relative_to(root))
        if relative in documents:
            return relative
        document, document_sha = read_json(path)
        row = dict(path=relative, sha256=document_sha, status=document.get('status'),
                   configuration={k: document[k] for k in CONFIG_KEYS if k in document},
                   model_identity=document.get('model_identity'),
                   source_claims=[], protocols=[])
        for name, expected in document.get('source_sha256', {}).items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise ValueError('Invalid recorded source identity')
            key = (name, expected)
            if key not in claims:
                claims[key] = observe_claim(root, name, expected)
                claims[key]['declared_by'] = []
            claims[key]['declared_by'].append(relative)
            row['source_claims'].append(dict(path=name, sha256=expected))
        documents[relative] = row
        protocol_sha = document.get('protocol_sha256')
        if isinstance(protocol_sha, str):
            matches = index.get(protocol_sha, [])
            row['protocol_sha256'] = protocol_sha
            row['protocol_status'] = 'exact' if matches else 'not_located_in_declared_corpus'
            row['protocols'] = [inspect(match) for match in matches]
            if any(documents[p]['sha256'] != protocol_sha for p in row['protocols']):
                row['protocol_status'] = 'changed_after_index'
        return relative

    bound = []
    for entry in variants:
        sources = entry['historical_evidence_routes']['sources']
        evidence = []
        for name in sources:
            path = local_path(root, str(Path('docs/plans') / name))
            if path.is_file():
                evidence.append(dict(path=inspect(path), status='located'))
            else:
                evidence.append(dict(path=str(path.relative_to(root)), status='missing'))
        bound.append(dict(item_id=entry['item_id'], variant=entry['record']['id'],
                          device=entry['record'].get('device'), evidence=evidence,
                          execution_adapter_review='pending'))
    return dict(
        schema='rheon.full-state-evidence-availability.v1',
        scope='Historical declaration/byte availability only; not fresh inference, timing or complete adapter qualification.',
        worklist=dict(path=str(worklist_path.relative_to(root)), sha256=worklist_sha),
        variants=bound, documents=list(documents.values()),
        source_claims=sorted(claims.values(), key=lambda row: (row['path'], row['expected_sha256'])),
        model_forwards=0, parameter_fits=0, complete_portfolio_benchmark=False,
    )


def summarize(result):
    claims = result['source_claims']
    documents = result['documents']
    return dict(
        schema=result['schema'], scope=result['scope'],
        variants=len(result['variants']),
        variants_with_all_evidence_located=sum(
            bool(v['evidence']) and all(e['status'] == 'located' for e in v['evidence'])
            for v in result['variants']),
        evidence_and_protocol_documents=len(documents),
        source_claims=len(claims), source_status=dict(Counter(c['status'] for c in claims)),
        protocol_status=dict(Counter(d['protocol_status'] for d in documents if 'protocol_status' in d)),
        conflicting_source_paths=sum(count > 1 for count in Counter(c['path'] for c in claims).values()),
        unresolved_source_claims=[c for c in claims if c['status'] != 'exact'],
        missing_protocol_bindings=[dict(path=d['path'], sha256=d['protocol_sha256'])
                                   for d in documents if d.get('protocol_status') != 'exact'
                                   and 'protocol_sha256' in d],
        model_forwards=0, parameter_fits=0, complete_execution_bindings=False,
        complete_portfolio_benchmark=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worklist', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise ValueError('Fresh audit result and summary paths required')
    result = audit(ROOT, args.worklist.resolve(), DOCS)
    summary = summarize(result)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    summary['result'] = dict(path=str(args.output.resolve().relative_to(ROOT)),
                             sha256=digest(args.output))
    summary['audit_source_sha256'] = digest(Path(__file__))
    with args.summary.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('unresolved_source_claims', 'missing_protocol_bindings')}))
