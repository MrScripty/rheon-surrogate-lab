"""Assemble retained read responses, preserving incomplete observations.

This reader never contacts Tuldok. In particular, a failed expensive Run listing
does not discard already captured models, materials, Studies or Trial lineage.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(capture, output):
    if output.exists():
        raise ValueError('Fresh assembled output required')
    rows = dict(models=[], study_summaries=[], study_pages=[], trials=[], lineage=[],
                recipe_materials=[], model_materials=[], project_snapshots=[], recipes=[],
                runs=[], projects=[], graphs=[])
    mapping = {
        'models.list': ('models', 'models'),
        'studies.list': ('study_summaries', 'studies'),
        'studies.get': ('study_pages', None),
        'studies.trial': ('trials', None),
        'studies.lineage': ('lineage', 'nodes'),
        'recipes.material': ('recipe_materials', None),
        'models.material': ('model_materials', None),
        'recipes.list_drafts': ('recipes', 'recipes'),
        'project_snapshots.list': ('project_snapshots', 'projects'),
        'runs.list': ('runs', 'self'),
        'projects.list': ('projects', 'self'),
        'graphs.list': ('graphs', 'self'),
    }
    sources, missing, responses = [], [], Counter()
    collection_tails, lineage_revisions = {}, set()
    for request_path in sorted((capture / 'calls').glob('*.request.json')):
        request = json.loads(request_path.read_text())
        response_path = request_path.with_name(request_path.name.replace('request.json', 'response.json'))
        if not response_path.exists():
            missing.append(dict(operation=request['operation'], request_path=str(request_path),
                                reason='No response retained; no server-cessation claim'))
            continue
        response = json.loads(response_path.read_text())
        operation = request['operation']
        key, field = mapping[operation]
        if field == 'self':
            rows[key].extend(response)
        elif field is None:
            rows[key].append(response)
        else:
            rows[key].extend(response[field])
        if field not in (None, 'self'):
            collection_tails[operation] = response['next_cursor']
        if operation == 'studies.lineage':
            lineage_revisions.add(response['revision'])
        responses[operation] += 1
        sources.append(dict(request=str(request_path), request_sha256=sha(request_path),
                            response=str(response_path), response_sha256=sha(response_path)))
    if any(v is not None for v in collection_tails.values()):
        raise ValueError('A paginated metadata collection was not completely observed')
    if len(lineage_revisions) != 1:
        raise ValueError('One revision-pinned lineage observation required')
    for field in ('models', 'study_summaries', 'trials', 'project_snapshots'):
        ids = [r['id'] for r in rows[field]]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate identity in captured collection: ' + field)
    trials = {r['id'] for r in rows['trials']}
    if trials != {r['trial']['id'] for r in rows['lineage']}:
        raise ValueError('Captured Trials and lineage membership disagree')
    if sum(r['trial_count'] for r in rows['study_summaries']) != len(trials):
        raise ValueError('Captured Study membership does not account for all Trials')
    run_summaries = {}
    for node in rows['lineage']:
        for run in node['runs']:
            if run['id'] in run_summaries and run_summaries[run['id']] != run:
                raise ValueError('Inconsistent repeated Run summary')
            run_summaries[run['id']] = run
    result = dict(schema='rheon.retained-catalog-observation.v1', created_unix=time.time(),
                  status='partial' if missing else 'captured',
                  captured_metadata=rows, lineage_revision=next(iter(lineage_revisions)),
                  run_summaries=list(run_summaries.values()),
                  full_run_collection_captured=responses['runs.list'] > 0,
                  missing_responses=missing, sources=sources,
                  source_sha256=sha(Path(__file__)),
                  limitations=['Metadata is not an artifact-verification certificate or a database backup.',
                               'Lineage Run summaries cover typed Trials only; they are not the complete Run collection.',
                               'No response is inferred for the timed-out collection request.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    summary = dict(schema=result['schema'], status=result['status'], output=str(output),
                   output_sha256=sha(output), counts={k: len(v) for k, v in rows.items()},
                   typed_trial_run_summaries=len(run_summaries),
                   lineage_records_authored=sum(n['lineage'] is not None for n in rows['lineage']),
                   lineage_run_summaries_truncated=sum(n['runs_truncated'] for n in rows['lineage']),
                   full_run_collection_captured=result['full_run_collection_captured'],
                   missing_operations=[r['operation'] for r in missing], read_responses=len(sources))
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    summary = assemble(args.capture, args.output)
    with args.summary.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
