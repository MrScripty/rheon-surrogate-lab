"""The benchmark's own result document: one JSON file per run.

It records what was measured (suite, cases, metrics), what was scored (subjects),
every value, and the Git commit of the code that produced it. Tuldok packets are
derived from it by packet.py; nothing else is written.

    {
      "format": "rheon_surrogate_benchmark_v1",
      "git_commit": "...", "git_dirty": false,
      "environment": {...},
      "suite":    {"id", "label", "population", "description", ...},
      "metrics":  [{"id", "label", "units", "description"}],
      "cases":    [{"id", <case attributes>}],
      "subjects": [{"id", "label", <subject attributes>}],
      "results":  [{"subject", "case", "values": {metric_id: number | null}, "failure": null | "reason"}]
    }

A null value with a null failure means the metric does not apply to that subject
and case; a failure reason means the run was attempted and did not finish.
"""
import json
import math
import re
import subprocess
from pathlib import Path

FORMAT = 'rheon_surrogate_benchmark_v1'
POPULATIONS = ('training', 'consumed', 'independent', 'reserved', 'unknown')
REPOSITORY = Path(__file__).resolve().parents[2]
# Tuldok's identifier rule, shortened so "<row>.<metric id>" observation ids still fit it.
MAX_ID_LENGTH = 96
IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,%d}$' % (MAX_ID_LENGTH - 1))


def git_identity(repository=REPOSITORY):
    """The commit a result was produced from, and whether the tree had uncommitted changes."""
    def git(*arguments):
        return subprocess.run(('git', *arguments), cwd=repository, check=True, capture_output=True,
                              text=True).stdout.strip()
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain'))}


def validate(document):
    """Raise ValueError unless `document` is a complete, self-consistent result document."""
    if document.get('format') != FORMAT:
        raise ValueError(f'Not a {FORMAT} document.')
    if document['suite']['population'] not in POPULATIONS:
        raise ValueError(f"Unknown population {document['suite']['population']!r}.")
    _unique_ids([document['suite']], 'suite')
    metric_ids = _unique_ids(document['metrics'], 'metric')
    for metric in document['metrics']:
        if not metric['label'].strip() or not metric['units'].strip():
            raise ValueError(f"Metric {metric['id']} needs a label and units (use 'ratio' or 'count' if it has none).")
    case_ids = _unique_ids(document['cases'], 'case')
    subject_ids = _unique_ids(document['subjects'], 'subject')
    seen = set()
    for result in document['results']:
        key = result['subject'], result['case']
        if result['subject'] not in subject_ids or result['case'] not in case_ids:
            raise ValueError(f'Result {key} names an undeclared subject or case.')
        if key in seen:
            raise ValueError(f'Result {key} appears twice.')
        seen.add(key)
        if set(result['values']) != metric_ids:
            raise ValueError(f'Result {key} does not list exactly the declared metrics.')
        for metric, value in result['values'].items():
            if value is not None and not (isinstance(value, (int, float)) and not isinstance(value, bool)
                                          and math.isfinite(value)):
                raise ValueError(f'Result {key} has a non-finite {metric}; record a failure instead.')
    missing = {(s, c) for s in subject_ids for c in case_ids} - seen
    if missing:
        raise ValueError(f'{len(missing)} subject/case pairs have no result, e.g. {sorted(missing)[0]}.')
    return document


def _unique_ids(entries, kind):
    identities = [entry['id'] for entry in entries]
    if len(set(identities)) != len(identities) or not identities:
        raise ValueError(f'Every {kind} needs a unique id and there must be at least one.')
    for identity in identities:
        if not IDENTIFIER.fullmatch(identity):
            raise ValueError(f'{kind} id {identity!r} must be letters, digits and _ . : - only, at most '
                             f'{MAX_ID_LENGTH} characters, starting with a letter or digit.')
    return set(identities)


def write(document, path):
    validate(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, allow_nan=False) + '\n')
    return path


def read(path):
    return validate(json.loads(Path(path).read_text()))
