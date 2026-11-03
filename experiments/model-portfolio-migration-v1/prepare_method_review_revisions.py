"""Prepare guarded authored method revisions without editing a native project.

The combined project's initial custody/cold/UI acceptance precedes adoption.
These outputs preserve original receipt/configuration suffixes and distinguish
authored interpretation from original scientific observations.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/model-portfolio-migration-v1'
PROJECT = RUN / 'native-composition-primary-01/root/project'
REVIEW = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/core-component-method-reviews-01.json'
CHECKPOINT = '0775213f6a11d93f9d80db80ea76daa6e80e2fcd0194a673dc2e8d093d644570'
HEADING = b'## Retained facts and comparison evidence\n\n'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')


def revised_card(raw, review):
    if raw.count(HEADING) != 1 or not raw.startswith(b'# '):
        raise ValueError('Expected the original captured card structure')
    title = raw.split(b'\n', 1)[0]
    prefix = title + ('\n\nRetrospective source review · '+review['fit_item_id']+
        '\n\nAuthored interpretation of captured source and receipts. '
        'No new training, benchmark, lineage or scientific promotion is implied.\n\n').encode('utf-8')
    for key, label in (('method','Method'),('purpose','Purpose'),('experiment','Experiment'),
                       ('outcome','Observed outcome'),('lesson','Lesson'),('reproduction','Reproduction')):
        if not isinstance(review[key], str) or not review[key].strip():
            raise ValueError('Complete authored review paragraphs required')
        prefix += ('**'+label+'.** '+review[key]+'\n\n').encode('utf-8')
    evidence = {k:review[k] for k in ('original_receipt','history','sources','source_witness',
        'training_population','training_inputs','data_generation_source') if k in review}
    prefix += b'## Exact source bindings for this review\n\n```json\n'+encode(evidence)+b'```\n\n'
    suffix = raw.split(HEADING, 1)[1]
    result = prefix+b'## Original retained fit facts before this review\n\n'+suffix
    return result, suffix


def verified_review_receipt(review):
    receipt = (ROOT/review['original_receipt']['path']).read_bytes()
    history = (ROOT/review['history']['path']).read_bytes()
    if sha(receipt) != review['original_receipt']['sha256'] or sha(history) != review['history']['sha256']:
        raise ValueError('Reviewed original receipt/history changed')
    original = json.loads(receipt)
    if original['outputs']['history.json']['sha256'] != sha(history):
        raise ValueError('History is not the recorded output of this training attempt')
    if 'source_witness' in review:
        selected = review['source_witness']; raw = (ROOT/selected['path']).read_bytes()
        if sha(raw) != selected['sha256'] or original['closure']['source_witness_sha256'] != sha(raw):
            raise ValueError('Source witness is not bound to this original receipt')
        bindings = json.loads(raw)['sources']
    else:
        bindings = original['closure']['sources']
    for source in review['sources']:
        path = ROOT/source['path']
        if sha(path.read_bytes()) != source['sha256'] or bindings[path.name] != source['sha256']:
            raise ValueError('Reviewed source differs from its exact original training closure')
    if 'training_population' in review:
        selected=review['training_population'];raw=(ROOT/selected['path']).read_bytes()
        if sha(raw)!=selected['sha256'] or original['outputs']['splits.json']['sha256']!=sha(raw):
            raise ValueError('Training population is not a recorded output of this fit')
    inputs={}
    for selected in review.get('training_inputs',[]):
        slot=selected['slot'];raw=(ROOT/selected['path']).read_bytes()
        expected=dict(sha256=sha(raw),bytes=len(raw))
        if (slot in inputs or selected['sha256']!=expected['sha256']
                or type(selected['bytes']) is not int or selected['bytes']!=len(raw)
                or original['closure'].get('inputs',{}).get(slot)!=expected):
            raise ValueError('Training input is not the exact recorded input slot of this fit')
        inputs[slot]=raw
    if 'data_generation_source' in review:
        selected=review['data_generation_source'];raw=(ROOT/selected['path']).read_bytes()
        manifest=inputs.get(selected['manifest_slot'])
        if (manifest is None or sha(raw)!=selected['sha256']
                or json.loads(manifest).get('source_sha256')!=sha(raw)):
            raise ValueError('Data-generation source differs from the original training manifest')
    return receipt


def prepare(output, review_path=REVIEW, *, project=None, checkpoint=None):
    if (project is None) != (checkpoint is None):
        raise ValueError('Select both source project and pinned checkpoint')
    project = PROJECT if project is None else project.resolve()
    checkpoint = CHECKPOINT if checkpoint is None else checkpoint
    if (not project.is_relative_to(ROOT) or type(checkpoint) is not str
            or len(checkpoint) != 64 or any(c not in '0123456789abcdef' for c in checkpoint)):
        raise ValueError('Task-local project and exact checkpoint required')
    if output.exists() or not output.is_relative_to(RUN):
        raise ValueError('Fresh task-owned preparation directory required')
    manifest_raw = (project/'.tuldok/project.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest['checkpoint'] != checkpoint:
        raise ValueError('Pinned source checkpoint changed')
    captured = {}
    for entry in manifest['inventory']:
        if entry['family'] != 'project_file':
            continue
        raw = (project/'.tuldok'/entry['path']).read_bytes()
        if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
            raise ValueError('Canonical authored file differs from its inventory')
        document = json.loads(raw)
        if document['project_id'] == manifest['project_id']:
            if document['path'] in captured:
                raise ValueError('Authored path repeats')
            captured[document['path']] = document
    review_raw = review_path.read_bytes()
    document=json.loads(review_raw);reviews = document['rows']
    index_path = 'notes/portfolio-index.json'
    index = json.loads((project/index_path).read_bytes())
    cards={fit['item_id']:fit['card_path'] for fit in index['fits']}
    if not 1<=len(reviews)<=38 or len({r['fit_item_id'] for r in reviews})!=len(reviews):
        raise ValueError('One bounded set of unique reviewed fits required')
    if any(cards.get(r['fit_item_id'])!=r['note_path'] for r in reviews):
        raise ValueError('Review does not select its exact original fit card')
    selected = {r['fit_item_id'] for r in reviews}
    if any(fit['item_id'] in selected and fit.get('method_review_status') == 'captured_source_and_receipt_reviewed'
           for fit in index['fits']):
        raise ValueError('Incremental preparation cannot replace an already reviewed fit')
    output.mkdir()
    changes = []

    def write(path, raw, **extra):
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'notes':
            raise ValueError('Authored path escapes the native notes directory')
        previous = (project/relative).read_bytes()
        current = captured[path]['current']
        if previous != current['text'].encode('utf-8') or sha(previous) != current['sha256']:
            raise ValueError('Frozen authored file differs from its captured content')
        if len(raw) > 1048576:
            raise ValueError('Native authored file size limit exceeded')
        raw.decode('utf-8')
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        changes.append(dict(path=path, kind='note', previous_sha256=sha(previous),
            expected_revision=current['revision'], revision=current['revision']+1,
            sha256=sha(raw), bytes=len(raw), **extra))

    for review in reviews:
        receipt = verified_review_receipt(review)
        path = review['note_path']
        old = (project/path).read_bytes()
        if old.count(receipt) != 1:
            raise ValueError('Card must contain the exact original receipt once')
        raw, suffix = revised_card(old, review)
        if raw.count(receipt) != 1 or not raw.endswith(suffix):
            raise ValueError('Original receipt/configuration evidence changed')
        write(path, raw, fit_item_id=review['fit_item_id'], original_suffix_sha256=sha(suffix),
              original_receipt_sha256=sha(receipt), original_receipt_bytes=len(receipt))
    edited = {n['fit_item_id']:n for n in changes}
    for fit in index['fits']:
        if fit['item_id'] in edited:
            change = edited[fit['item_id']]
            fit.update(card_sha256=change['sha256'], method_review_status='captured_source_and_receipt_reviewed',
                       clean_reproduction_qualified=False)
    current_review = dict(path=str(review_path.relative_to(ROOT)), sha256=sha(review_raw))
    batches = index['sources'].get('method_review_batches', [])
    if type(batches) is not list:
        raise ValueError('Prior review-batch provenance must be a list')
    batches = list(batches)
    for source in (index['sources'].get('method_reviews'), current_review):
        if source is not None and source not in batches:
            batches.append(source)
    index['sources']['method_reviews'] = current_review
    index['sources']['method_review_batches'] = batches
    write(index_path, encode(index))
    overview_path = 'notes/portfolio-overview.md'
    overview = (project/overview_path).read_bytes()
    addition = b'''\n## How the expanded generator's components differ\n\nThe three core component cards now distinguish method, purpose, training evidence,\nobserved outcome and lessons. Their interpretation is authored retrospectively;\noriginal receipts/configurations remain exact below each review.\n\nThe shared spectral pressure response has four learned float64 coefficients fitted\non148training scenes. The local sampler has48learned float64 parameters fitted\non16scenes. Its prescribed interpolation graph remains part of the method. The\nfixed-viscosity parent records100456total parameters, of which6600were trainable;\nits243-coefficient calibrated viscosity convolution was frozen. These counts and\ntraining populations belong to different components. A later composition may\nreplace components and cannot inherit the full parent's executed parameter count.\n\nTiny component training objectives are different from pressure-field error,\nequation residual and own-history drift. The quoted full-state accuracy anchors\nreproduced on their original consumed cases, with their original integration work.\nThe N8-labelled cell-coordinate anchor actually takes its N1 branch on the quoted\ncase. Existing K-output generators still advance each physical output in sequence;\nthese components do not establish endpoint prediction with cheap intermediate\nframes. Fresh common complete-generator timing and clean retraining remain pending.\n'''
    if 'overview_addition' in document:
        if not isinstance(document['overview_addition'],str) or not document['overview_addition'].strip():
            raise ValueError('Authored overview addition must contain review context')
        addition=document['overview_addition'].encode('utf-8')
    write(overview_path, overview+addition)
    if (project/'.tuldok/project.json').read_bytes() != manifest_raw:
        raise ValueError('Primary source checkpoint changed during preparation')
    result = dict(schema='rheon.method-review-note-revisions.v1', status='prepared_not_published',
        source_project_id=manifest['project_id'], source_checkpoint=checkpoint,
        source_project_path=str(project.relative_to(ROOT)),
        source_manifest_sha256=sha(manifest_raw), review_path=str(review_path.relative_to(ROOT)),
        review_sha256=sha(review_raw), notes=changes, original_receipt_suffixes_preserved=len(reviews),
        new_neural_forwards=0, native_mutations=0, clean_reproduction_qualified=False,
        adoption_precondition='First qualify final combined custody/cold/UI; then verify target file hashes/revisions and obtain its actual binding/checkpoint before editing/adopting.')
    (output/'preparation.json').write_bytes(encode(result))
    (output/'adopt-items.json').write_bytes(encode(dict(items=[
        {k:n[k] for k in ('kind','path','expected_revision')} | dict(expected_sha256=n['sha256']) for n in changes])))
    print(json.dumps(dict(status=result['status'], reviewed_cards=len(reviews), changed_notes=len(changes),
                          adoption_bytes=sum(n['bytes'] for n in changes), original_source_unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reviews',type=Path,default=REVIEW)
    parser.add_argument('--project',type=Path,help='Explicit task-local source project; pair with --checkpoint')
    parser.add_argument('--checkpoint',help='Exact current canonical checkpoint of the selected source project')
    args = parser.parse_args()
    prepare(args.output.resolve(),args.reviews.resolve(),
            project=args.project.resolve() if args.project else None,checkpoint=args.checkpoint)
