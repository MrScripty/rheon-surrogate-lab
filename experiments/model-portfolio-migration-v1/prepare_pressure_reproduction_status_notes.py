"""Correct current qualification wording;preserve old statements and metrics."""
from collections import Counter
import copy
from pathlib import Path
import re
import textwrap
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,header,read,save,binding,digest,encode,require
from adopt_matched_generator_outcome_notes_02 import target_records
from prepare_matched_generator_outcome_notes import replace_paragraph,TAIL

OUTPUT = RUN/'native-pressure-reproduction-status-notes-01'
REPLACEMENTS = (
    ('The historical trainer/restore/publication invokes CPU neural inference; qualify GPU-only equivalents before replay.',
     'The historical training and publication paths still need GPU-only replay qualification; the separate original-checkpoint GPU evaluation and timing are now qualified.'),
    ('Historical publication/restore contains CPU neural queries; qualify a GPU-only equivalent before replay.',
     'Historical training and publication replay still needs a GPU-only route; the separate original-checkpoint GPU evaluation and timing are now qualified.'),
    ('The historical trainer performs CPU publication/restore neural forwards; qualify a GPU-only route before rerunning under current policy.',
     'The historical trainer performs CPU publication/restore neural forwards and needs a GPU-only route before replay; the separate original-checkpoint GPU evaluation and timing are now qualified.'),
    ('Clean retraining, new-Rheon regeneration and a current common benchmark for this unnamed fit remain unqualified.',
     'Clean retraining and new-Rheon regeneration remain unqualified; the current common GPU pressure accuracy and latency are qualified.'),
    ('Clean retraining, fresh qualified latency and new-Rheon data equivalence remain unqualified; no checkpoint can be retired on this review alone.',
     'Clean retraining and new-Rheon data equivalence remain unqualified; the current common GPU pressure accuracy and latency are qualified. These results alone do not authorize checkpoint retirement.'),
)


def corrected_reproduction(text):
    counts = {}
    for old,new in REPLACEMENTS:
        require(text.count(old) <= 1,'Repeated original status sentence')
        if old in text:
            counts[old] = 1
            text = text.replace(old,new,1)
    require(counts and 'a current common benchmark for this unnamed fit remain unqualified' not in text
            and 'fresh qualified latency' not in text,'Current status correction did not resolve stale scope')
    return text,counts


def reflow_prose(text):
    """Preserve Markdown words and structured lines;only prose whitespace changes."""
    result,paragraph,fenced = [],[],False
    def flush():
        if paragraph:
            result.append(textwrap.fill(' '.join(paragraph),width=100,break_long_words=False,break_on_hyphens=False))
            paragraph.clear()
    for line in text.splitlines():
        if line.startswith('```'):
            flush();fenced = not fenced;result.append(line)
        elif fenced or not line.strip() or line.startswith(('#','|','- ','* ')):
            flush();result.append(line)
        else:
            paragraph.append(line)
    flush()
    proposed = '\n'.join(result)+'\n'
    require(re.sub(r'\s+',' ',proposed).strip() == re.sub(r'\s+',' ',text).strip(),'Overview words or scientific values changed')
    return proposed


def run():
    require(not OUTPUT.exists(),'Fresh status correction proposal required')
    custody_path = DOC/'reports/native-pressure-outcome-adoption-01.json'
    custody = read(custody_path)
    browser_path = RUN/'native-pressure-publication-browser-01/report.json'
    browser = read(browser_path)
    require(custody['status'] == 'native26pressure_outcome_notes_exact_and_scientific_records_unchanged'
            and browser['status'] == 'passed_native_project_UI_actions' and browser['browser_stopped']
            and len(browser['steps']) == 1666 and len(browser['api_requests']) == 54,'Original native notes/custody/actual browser must finish first')
    _,snapshot = header()
    require(snapshot['checkpoint'] == custody['checkpoint'],'Current accepted note checkpoint changed')
    current = target_records(snapshot)
    guide_path,index_path = 'notes/00-models.portfolio.json','notes/portfolio-index.json'
    guide,index = read(PROJECT/guide_path),read(PROJECT/index_path)
    nodes = {node['id']:node for node in guide['models']}
    fits = {fit['item_id']:fit for fit in index['fits']}
    prior_proposal_path = RUN/'native-pressure-outcome-notes-01/preparation.json'
    prior = read(prior_proposal_path)
    targets = [note for note in prior['notes'] if note.get('fit_item_id')]
    require(len(targets) == 23 and len(nodes) == len(fits) == 205,'Exact23target/205organization identities required')
    staged,notes,counts = {},[],Counter()

    def put(path,raw,**extra):
        entry,doc = current[path]
        old = doc['current']
        require((PROJECT/path).read_bytes() == old['text'].encode() and digest(old['text'].encode()) == old['sha256']
                and 0 < len(raw) <= 1048576,'Current/proposed note differs or exceeds bound')
        staged[path] = raw
        notes.append(dict(kind='note',path=path,expected_revision=old['revision'],revision=old['revision']+1,
            previous_sha256=old['sha256'],sha256=digest(raw),bytes=len(raw),file_id=entry['id'],
            prepared_path=str((OUTPUT/path).relative_to(ROOT)),**extra))

    for previous in targets:
        fid = previous['fit_item_id']
        fit,node = fits[fid],nodes[fid]
        original = (PROJECT/fit['card_path']).read_text()
        old = node['summary']['reproduction']
        new,replaced = corrected_reproduction(old)
        counts.update(replaced)
        proposed = replace_paragraph(original,'reproduction',old,new)
        archive = ('## Superseded benchmark-stage reproduction statement\n\nThe current paragraph above corrects its qualification status. '
            'This original statement is retained for history.\n\n```json\n'+encode(old).decode()+'```\n\n')
        require(proposed.count(TAIL) == 1,'Original receipt/configuration marker differs')
        proposed = proposed.replace(TAIL,archive+TAIL,1)
        tail = original.split(TAIL,1)[1].encode()
        receipt = (ROOT/previous['original_receipt']['path']).read_bytes()
        require(proposed.encode().endswith(tail) and original.encode().count(receipt) == proposed.encode().count(receipt) == 1,
                'Original receipt/configuration tail changed')
        put(fit['card_path'],proposed.encode(),fit_item_id=fid,original_tail_sha256=digest(tail),
            original_receipt=previous['original_receipt'])
        node['summary']['reproduction'] = new
        node['card'].update(revision=notes[-1]['revision'],sha256=notes[-1]['sha256'])
        fit['card_sha256'] = notes[-1]['sha256']
    require(sum(counts.values()) == 36 and sorted(counts.values()) == [1,3,6,13,13],
            'Expected exact23current paragraphs and14explicit stale-status clauses')
    put(index_path,encode(index))
    put(guide_path,encode(guide))
    overview_path = 'notes/portfolio-overview.md'
    overview = reflow_prose((PROJECT/overview_path).read_text())
    put(overview_path,overview.encode())
    require(len(notes) == 26 and sum(note['bytes'] for note in notes) <= 8*1048576,'Native26note batch bound exceeded')
    _,latest = header()
    require(latest['checkpoint'] == snapshot['checkpoint'],'Current native project changed during proposal')
    OUTPUT.mkdir()
    for path,raw in staged.items():
        target = OUTPUT/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    save(OUTPUT/'preparation.json',dict(status='pressure26outcome_notes_prepared_not_adopted',created_unix=time.time(),
        source_project_id=snapshot['project_id'],source_checkpoint_at_read=snapshot['checkpoint'],notes=notes,
        pressure_fits=23,models=205,authored_comparisons=26,guide_revision=current[guide_path][1]['current']['revision']+1,
        literal_summary_paragraphs=1230,original_receipt_configuration_tails_preserved=23,previous_review_statements_retained=True,
        correction='I57qualification status only;all method,purpose,experiment,outcome,lesson,metric,measurement source and fit facts unchanged. Overview reflows identical words.',
        status_sentence_replacements=dict(counts),producer=binding(Path(__file__)),
        evidence=[binding(path) for path in (custody_path,browser_path,prior_proposal_path,
            RUN/'native-pressure-publication-browser-01/visual-inspection.json',DOC/'reports/original-pressure-GPU-timing-batch-01.json',
            DOC/'reports/all150-pressure-GPU-measurements-qualification-01.json')],
        native_mutations=0,new_model_forwards=0,Trial_lineage_modified=False,full_portfolio_complete=False))
    print('Prepared26status corrections:23coherent reproduction paragraphs andword-identical overview reflow;all metrics unchanged.',flush=True)


if __name__ == '__main__':
    run()
