"""Join two source-reviewed families for one bounded native note publication."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = DOC/'reports/captured-operator-families-method-reviews-01.json'


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def combine():
    if OUTPUT.exists():
        raise ValueError('Preserve original combined review')
    inputs = [DOC/'reports'/name for name in (
        'captured-residual-stability-method-reviews-01.json',
        'captured-multigrid-recurrence-method-reviews-01.json')]
    reports = [json.loads(path.read_bytes()) for path in inputs]
    rows = [row for report in reports for row in report['rows']]
    if (len(rows) != 21 or len({r['fit_item_id'] for r in rows}) != 21
            or {r['named_checkpoint'] for r in rows} != {f'C{n:02}' for n in range(16,37)}):
        raise ValueError('Exactly the two disjoint C16--C36 reviewed families required')
    helpers = []
    for report in reports:
        for spec in [report['producer'],*report['helpers']]:
            if binding(ROOT/spec['path']) != spec:
                raise ValueError('Source-review producer/helper changed')
            if spec not in helpers:
                helpers.append(spec)
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=21,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,source_review_reports=[binding(p) for p in inputs],
        overview_addition=''.join(report['overview_addition'] for report in reports),
        producer=binding(Path(__file__)),helpers=helpers+[binding(p) for p in inputs])
    OUTPUT.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print('21 source-reviewed fits combined; no native mutation.')


if __name__ == '__main__':
    combine()
