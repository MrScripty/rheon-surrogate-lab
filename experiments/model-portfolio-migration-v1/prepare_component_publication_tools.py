"""Bind a new 29-review publication to the previously accepted note workflow."""
import ast
from pathlib import Path
import re
import time

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/model-portfolio-migration-v1'
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
JOB = RUN / 'method-review-native-adoption-14'
REPORT = DOC / 'reports/component-publication-tool-preparation-01.json'


def change(text, old, new):
    require(old in text, 'Required accepted-template adaptation absent: ' + old)
    return text.replace(old,new)


def numbers(text, replacements):
    return re.sub(r'\b(?:' + '|'.join(map(str,replacements)) + r')\b',
                  lambda match:str(replacements[int(match[0])]),text)


def run():
    targets = [HERE / n for n in ['prepare_pending_component_publication.py',
        'qualify_pending_component_publication.cjs','prepare_component_review_browser.py']]
    require(not REPORT.exists() and not JOB.exists() and not any(p.exists() for p in targets),
            'Preserve original component publication tools')
    accepted = read(DOC / 'reports/generator-native-owner-preparation-01.json')
    original_job = RUN / 'method-review-native-adoption-13'
    require([binding(original_job/n) for n in ['run.py','audit.py']] == accepted['current'],
            'Previously accepted native owners changed')
    custody = read(DOC / 'reports/native-116-method-review-adoption-01.json')
    require(custody['reviewed_fit_cards'] == 116 and custody['checkpoint'] ==
            '60bd3ec54bd2832177c951a53bee35f3f9b5a90883c1bd195dc0e7e430fb8900',
            'Current native acceptance differs')
    old_prep = HERE / 'prepare_pending_generator_publication.py'
    prep = old_prep.read_text()
    for old,new in [
        ('native-generator-review-publication-01','native-component-review-publication-01'),
        ('captured-pending-generator-combined-reviews-01.json','captured-pending-component-combined-reviews-01.json'),
        ('review-coverage-83.json','review-coverage-88.json'),
        ('48b888bb04e554f672e2983185e9c8a6a7069a100b2bd19ea659a19fcb7bf32a',custody['checkpoint']),
        ("coverage['source_reviewed'] == 116","coverage['source_reviewed'] == 145"),
        ("sum(bool(r.get('method_review_status')) for r in fits.values()) == 79",
         "sum(bool(r.get('method_review_status')) for r in fits.values()) == 116"),
        ("current[GUIDE][1]['current']['revision'] == 2","current[GUIDE][1]['current']['revision'] == 3"),
        ("guide['description'].count('79 have retrospective')","guide['description'].count('116 have retrospective')"),
        ("replace('79 have retrospective','116 have retrospective')","replace('116 have retrospective','145 have retrospective')"),
        ("for m in guide['models']) == 116","for m in guide['models']) == 145"),
        ('reviewed_total=116','reviewed_total=145'),('review_pending=89','review_pending=60'),
        ('guide_revision=3','guide_revision=4'),('literal_summary_paragraphs=696','literal_summary_paragraphs=870')]:
        prep = change(prep,old,new)
    # Only the remaining population literals change, never a hash substring.
    prep = numbers(prep,{37:29,40:32})
    prep = prep.replace('116 have retrospective source reviews','145 have retrospective source reviews')
    prep = prep.replace('Another 89 source reviews remain','Another 60 source reviews remain')
    marker = '\n## Work still required\n'
    extra = '''
## Pressure components and comparison scope

The newer component reviews distinguish modal refinements, local operator
calibration, causal request prediction, guided temporal corrections and
pressure replay/student/GCR experiments. Learned parameter counts exclude
frozen components and numerical work unless explicitly stated. The earliest
16-cubed models use anisotropic spacing and fluid density 1.25 kg/m3; later
cubic-cell water models generally use 1,000 kg/m3.

C112--C118 common GPU results measure their local learned pressure readout,
including its input/output transfers. They exclude the numerical hierarchy
and GCR composition used in the intended solver. Their timing and error cannot
be presented as a complete hybrid solve. The spectrum-augmentation fit adds
manufactured spatial pressure patterns, not simulated fluid trajectories.

Worst-first checkpoint selection can improve a dominant equation metric while
worsening pressure and average loss. The guided pressure fit retains its
selected probe, mean-best and last-evaluated states separately; they must not
be silently substituted when comparing results.
'''
    prep = change(prep,marker,extra+marker)
    qualifier_path = HERE / 'qualify_pending_generator_publication.cjs'
    qualifier = qualifier_path.read_text()
    qualifier = change(qualifier,'native-generator-review-publication-01','native-component-review-publication-01')
    qualifier = change(qualifier,'generator-review-native-preparation-01.json','component-review-native-preparation-01.json')
    qualifier = numbers(qualifier,{40:32,37:29,116:145,696:870,89:60})
    qualifier = change(qualifier,'prep.guide_revision,3','prep.guide_revision,4')
    qualifier = change(qualifier,'notes.get(guidePath).revision,3','notes.get(guidePath).revision,4')
    qualifier = change(qualifier,'guide_revision:3','guide_revision:4')
    qualifier = change(qualifier,'end).trim(),row[fields[i]]);','end).trim(),row[fields[i]].trim());')
    qualifier = change(qualifier,'new_model_forwards:0,payload_copies:0,producer:',
                       'paragraph_boundary_whitespace_normalized:true,new_model_forwards:0,payload_copies:0,producer:')
    qualifier = qualifier.replace('forty_note_packet','thirtytwo_note_packet')
    owner = (original_job / 'run.py').read_text()
    for old,new in [('native-generator-review-publication-01','native-component-review-publication-01'),
                    ('generator-review-native-preparation-01.json','component-review-native-preparation-01.json'),
                    ('thirty-seven','twenty-nine'),('forty','thirtytwo')]:
        owner = change(owner,old,new)
    owner = numbers(owner,{40:32,37:29,116:145,79:116})
    owner = change(owner,'native-postboot-project-browser-01','native-116-review-browser-01')
    audit = (original_job / 'audit.py').read_text()
    audit = change(audit,'native-generator-review-publication-01','native-component-review-publication-01')
    audit = numbers(audit,{40:32,116:145})
    audit = change(audit,'native-seventynine-method-review-adoption-01.json','native-116-method-review-adoption-01.json')
    audit = audit.replace('forty','thirtytwo').replace('Forty','Thirty-two')
    browser_path = HERE / 'prepare_generator_review_browser.py'
    browser = browser_path.read_text()
    browser = change(browser,"JOB = RUN/'method-review-native-adoption-13'","JOB = RUN/'method-review-native-adoption-14'")
    browser = change(browser,'native-generator-review-publication-01','native-component-review-publication-01')
    browser = numbers(browser,{40:32,37:29,116:145})
    browser = browser.replace('forty','thirtytwo')
    browser = change(browser,"text='revision 3'","text='revision 4'")
    for text in (prep,owner,audit,browser):
        ast.parse(text)
    originals = [binding(p) for p in [old_prep,qualifier_path,browser_path,original_job/'run.py',original_job/'audit.py']]
    for path,text in zip(targets,[prep,qualifier,browser],strict=True):
        path.write_text(text)
    JOB.mkdir()
    (JOB / 'run.py').write_text(owner)
    (JOB / 'audit.py').write_text(audit)
    result = dict(status='prepared_29review_32note_145total_tools_not_dispatched',created_unix=time.time(),
        originals=originals,current=[binding(p) for p in [*targets,JOB/'run.py',JOB/'audit.py']],
        expected_notes=32,new_reviewed_cards=29,total_reviewed_cards=145,guide_revision=4,
        expected_literal_paragraphs=870,native_requests_started=0,native_project_mutations=0,
        new_model_forwards=0,payload_copies=0,producer=binding(Path(__file__)))
    REPORT.write_bytes(canonical(result)+b'\n')
    print('29-review/32-note native preparation, independent qualifier, owner, audit and browser tools prepared.')


if __name__ == '__main__':
    run()
