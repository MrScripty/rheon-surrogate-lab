"""Prepare all37pending reviews and one coherent116-review native guide.

Read canonical notes in place; write only proposed text/JSON notes. No project,
weight, dataset, original receipt, scientific record or Trial lineage is changed.
"""
import copy
from pathlib import Path
import time

from captured_temporal_review import verified_review_receipt as verify_temporal
from prepare_authored_model_portfolio import summary
from prepare_method_review_revisions import encode, revised_card, verified_review_receipt as verify_legacy
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
RUN = ROOT/'runs/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = RUN/'native-generator-review-publication-01'
COMBINED = DOC/'reports/captured-pending-generator-combined-reviews-01.json'
GUIDE = 'notes/00-models.portfolio.json'
OVERVIEW = '''# Rheon model portfolio

This project retains 205 recorded fits/components and 524 parameter byte states.
Those are different counts. The model guide groups all fits by task and method;
116 have retrospective source reviews, including all 35 full-state generator
fits. Another 89 source reviews remain. Original receipts, configurations,
scientific history and unknown Trial ancestry remain preserved. Authored
comparisons describe evidence and do not invent historical Trial lineage.

## Read the comparisons by task

Pressure-only models return pressure for an externally supplied pressure request.
Their error and time are not complete simulation-step measurements. Complete
generators must return pressure, staggered velocities and tracer at every output.
For K outputs, report full block time and block/K; each output advances the whole
declared timestep. Grid dimensions, cells per metre, timestep, reference solver,
precision and fitting exposure remain separate comparison fields.

The original near-exact F64 anchors reproduce on their original consumed cases:
approximately 9.25e-17 pressure error at 0.23 ms/output for the cell sampler,
and 9.16e-16 at 0.41 ms/output for the temporal baseline. They use 15,360-cell
rectangles, different cadences and different internal work. They are not
universal full-model accuracy or independent holdout guarantees.

The 127 available named pressure fits have a common retained 128-request GPU
comparison. Four early direct joint generators also have a common 64-case,
full-state GPU comparison: K1/2/3/4 median ms/output 3.168/1.496/1.008/0.790,
with worst mean-free pressure relative L2 1.018/1.039/1.329/1.492. Their largest
tails occur at the initial state. Original saved CPU probes reproduce the first
case's error metrics without new CPU inference. These are 16 consumed fitting
scenes, not a holdout, and the population differs from the quoted F64 cases.
All new neural evaluation is CUDA-only; GPU timings are labeled as such.

## What changed between generator designs

Joint heads share an encoder and predict all future fields together. Endpoint
designs predict a final state and condition cheaper intermediate readouts on it.
Retained-feature designs update latent features instead of rerunning the main
encoder. Coupled and displacement designs share an encoder but still execute
pressure/transport readouts at every output. All 35 generator cards now explain
their actual branch, inherited coefficients, training data and limitations.

Early raw-tensor training feedback differs from deployed reconstruction/repacking.
Later fixed force/increment scales change units, not timestep coverage. Matched
MSE experiments separate field denominators from the earlier logarithmic loss.
The fixed-viscosity horizon studies balance 36,000 training predictions while
changing optimizer-update counts; repetitions change both model and sampling
seeds. The curl readout constrains ideal increments, not existing divergence or
all-field accuracy. A precise frozen pressure component can receive an inaccurate
predicted request and therefore produce an inaccurate future simulation state.

## Work still required

The six endpoint/reuse comparison is prepared on the exact same 64 cases as the
joint models. Their fresh results, and matching F64 baselines on that population,
remain pending. Full benchmarking of all fits/parameter states and own-history
drift are incomplete. Several inherited-pressure constructors execute neural
unit responses on CPU before moving to CUDA; qualify GPU-only setup variants
and retain precision differences before running them. Original recipes also
need GPU-only publication probes and clean reproduction qualification.

Retained old-Rheon references remain the comparison authority. The newer Rheon
changes physics and must remain a separate epoch; training/test regeneration
scripts for it are still required. Thirty-five parameter states remain outside
native declarations, and 936 registrations still need disposition. No unique
weights are approved for deletion from this accounting alone. Final portable
save/load and practical opening performance remain unfinished.
'''


def verified_receipt(row):
    contract = row.get('source_closure_semantics')
    if contract is None:
        return verify_legacy(row)
    require(contract == 'project_relative_paths_including_expert_assets', 'Unsupported authored source-closure contract')
    return verify_temporal(row)


def run():
    require(not OUTPUT.exists() and not COMBINED.exists(), 'Preserve original37review preparation')
    coverage_path = DOC/'reports/review-coverage-83.json'
    coverage = read(coverage_path)
    pending = {r['fit_item_id'] for r in coverage['unpublished']}
    require(len(pending) == 37 and coverage['source_reviewed'] == 116, 'Exact pending source-review population required')
    report_paths = sorted({p for r in coverage['unpublished'] for p in r['reports']})
    reports = [read(ROOT/p) for p in report_paths]
    rows = [r for report in reports for r in report['rows'] if r['fit_item_id'] in pending]
    require(len(rows) == len({r['fit_item_id'] for r in rows}) == 37, 'Distinct complete review rows required')
    for report in reports:
        for spec in [report['producer'],*report.get('helpers',[])]:
            require(binding(ROOT/spec['path']) == spec, 'Original source-review producer/helper changed')
    header_path = PROJECT/'.tuldok/project.json'
    header_binding = binding(header_path)
    header = read(header_path)
    require(header['checkpoint'] == '48b888bb04e554f672e2983185e9c8a6a7069a100b2bd19ea659a19fcb7bf32a',
            'Current canonical checkpoint differs')
    current = {}
    for entry in header['inventory']:
        if entry['family'] != 'project_file':
            continue
        p = PROJECT/'.tuldok'/entry['path']
        require(binding(p)['sha256'] == entry['sha256'] and p.stat().st_size == entry['bytes'], 'Canonical authored record differs')
        value = read(p)
        if value['project_id'] == header['project_id']:
            require(value['path'] not in current, 'Duplicate current authored path')
            current[value['path']] = (entry,value)
    combined = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=37,source_review_reports=[binding(ROOT/p) for p in report_paths],new_model_forwards=0,
        native_mutations=0,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_temporal_review.py','prepare_method_review_revisions.py','prepare_authored_model_portfolio.py']])
    COMBINED.write_bytes(canonical(combined)+b'\n')
    OUTPUT.mkdir()
    changes = []

    def put(path, raw, **extra):
        _,document = current[path]
        old = document['current']
        original = (PROJECT/path).read_bytes()
        require(original == old['text'].encode() and binding(PROJECT/path)['sha256'] == old['sha256']
                and len(raw) <= 1048576, 'Current/proposed authored file differs or exceeds native bound')
        raw.decode('utf-8')
        target = OUTPUT/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        changes.append(dict(kind='note',path=path,expected_revision=old['revision'],revision=old['revision']+1,
            previous_sha256=old['sha256'],sha256=binding(target)['sha256'],bytes=len(raw),
            prepared_path=str(target.relative_to(ROOT)),**extra))

    index = read(PROJECT/'notes/portfolio-index.json')
    fits = {r['item_id']:r for r in index['fits']}
    require(len(fits) == 205 and sum(bool(r.get('method_review_status')) for r in fits.values()) == 79,
            'Complete current index/79published reviews required')
    for row in rows:
        require(not fits[row['fit_item_id']].get('method_review_status') and fits[row['fit_item_id']]['card_path'] == row['note_path'],
                'Cannot replace an existing reviewed fit or another card')
        receipt = verified_receipt(row)
        old = (PROJECT/row['note_path']).read_bytes()
        raw,suffix = revised_card(old,row)
        require(old.count(receipt) == raw.count(receipt) == 1 and raw.endswith(suffix), 'Original receipt/config suffix changed')
        put(row['note_path'],raw,fit_item_id=row['fit_item_id'],original_suffix_sha256=__import__('hashlib').sha256(suffix).hexdigest(),
            original_receipt_sha256=__import__('hashlib').sha256(receipt).hexdigest(),original_receipt_bytes=len(receipt))
    edited = {n['fit_item_id']:n for n in changes}
    for fid,note in edited.items():
        fits[fid].update(card_sha256=note['sha256'],method_review_status='captured_source_and_receipt_reviewed')
    provenance = binding(COMBINED)
    batches = list(index['sources'].get('method_review_batches',[]))
    for spec in (index['sources'].get('method_reviews'),provenance):
        if spec is not None and spec not in batches:
            batches.append(spec)
    index['sources'].update(method_reviews=provenance,method_review_batches=batches)
    put('notes/portfolio-index.json',encode(index))
    put('notes/portfolio-overview.md',OVERVIEW.encode())
    guide = copy.deepcopy(read(PROJECT/GUIDE))
    require(current[GUIDE][1]['current']['revision'] == 2 and len(guide['relationships']) == 26, 'Exact guide revision2 required')
    models = {m['id']:m for m in guide['models']}
    require(models.keys() == fits.keys(), 'Guide/index fit membership differs')
    for fid,note in edited.items():
        m = models[fid]
        require(m['review'] == 'review_pending' and m['summary'] is None, 'Cannot replace previous guide summaries')
        m['review'] = 'source_reviewed'
        m['summary'] = summary((ROOT/note['prepared_path']).read_text(),fid)
        m['card'].update(revision=note['revision'],sha256=note['sha256'])
    require(guide['description'].count('79 have retrospective') == 1, 'Current guide description differs')
    guide['description'] = guide['description'].replace('79 have retrospective','116 have retrospective')
    require(sum(m['review'] == 'source_reviewed' for m in guide['models']) == 116, 'Complete reviewed-guide total differs')
    put(GUIDE,canonical(guide)+b'\n')
    require(len(changes) == 40 and len({n['path'] for n in changes}) == 40
            and binding(header_path) == header_binding, 'Exact40notes and unchanged canonical source required')
    packet = dict(status='prepared_not_published',created_unix=time.time(),source_project_id=header['project_id'],
        source_checkpoint=header['checkpoint'],source_manifest_sha256=header_binding['sha256'],source_project_path=str(PROJECT.relative_to(ROOT)),
        notes=changes,new_reviewed_fits=37,reviewed_total=116,models=205,review_pending=89,guide_revision=3,
        literal_summary_paragraphs=696,authored_comparisons=26,original_receipt_suffixes_preserved=37,
        new_model_forwards=0,native_mutations=0,payload_copies=0,Trial_lineage_modified=False,full_portfolio_complete=False,
        review_path=str(COMBINED.relative_to(ROOT)),review_sha256=provenance['sha256'],producer=binding(Path(__file__)),
        evidence=[binding(coverage_path),provenance,*[binding(ROOT/p) for p in report_paths],binding(PROJECT/GUIDE)])
    (OUTPUT/'preparation.json').write_bytes(canonical(packet)+b'\n')
    print('Prepared40notes/37new reviews/116total/696literal paragraphs; no native mutation or dataset/weight copy.')


if __name__ == '__main__':
    run()
