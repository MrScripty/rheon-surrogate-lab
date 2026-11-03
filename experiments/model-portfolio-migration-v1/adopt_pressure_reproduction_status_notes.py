"""A separate guarded26note correction using the qualified native note owner."""
import argparse
from pathlib import Path
import time

import adopt_pressure_outcome_notes as owner

OLD_CUSTODY = owner.REPORT
owner.PREPARED = owner.RUN/'native-pressure-reproduction-status-notes-01'
owner.QUALIFICATION = owner.DOC/'reports/pressure-reproduction-status-notes-qualification-01.json'
owner.JOB = owner.RUN/'native-pressure-reproduction-status-adoption-01'
owner.BEFORE = owner.JOB/'before-custody.json'
owner.STATUS = owner.JOB/'status.json'
owner.REQUEST = owner.JOB/'adopt.request.json'
owner.RESPONSE = owner.JOB/'adopt.response.json'
owner.REPORT = owner.DOC/'reports/native-pressure-reproduction-status-adoption-01.json'


def before():
    owner.require(not owner.JOB.exists() and not owner.REPORT.exists(),'Fresh correction owner required;never replay')
    prep,proof = owner.qualified()
    owner.require(proof['five_scientific_summary_fields_unchanged'] == 23
            and proof['current_reproduction_status_coherent'] and proof['overview_words_and_metric_values_unchanged'],
            'Only the qualified current-status/whitespace correction may be published')
    custody = owner.read(OLD_CUSTODY)
    prior = owner.read(owner.ROOT/custody['original_owner']['path'])
    browser = owner.read(owner.RUN/'native-pressure-publication-browser-01/report.json')
    owner.require(custody['status'] == 'native26pressure_outcome_notes_exact_and_scientific_records_unchanged'
            and owner.binding(owner.ROOT/custody['original_owner']['path']) == custody['original_owner']
            and prior['status'] == 'native_pressure26note_adoption_acknowledged'
            and not Path(f'/proc/{prior["pid"]}').exists() and browser['status'] == 'passed_native_project_UI_actions'
            and browser['browser_stopped'],'Original accepted publication and browser must be terminal')
    raw,snapshot = owner.header()
    owner.require(snapshot['checkpoint'] == custody['checkpoint'] == prior['checkpoint']
            and owner.digest(raw) == custody['canonical_header_sha256'] and owner.payloads() == custody['payload_preservation'],
            'Current project must equal prior exact note custody')
    current = owner.target_records(snapshot)
    targets = []
    for note in prep['notes']:
        entry,doc = current[note['path']]
        owner.require(entry['id'] == note['file_id'] and doc['current']['revision'] == note['expected_revision']
                and doc['current']['sha256'] == note['previous_sha256']
                and (owner.PROJECT/note['path']).read_bytes() == doc['current']['text'].encode(),
                'A target note changed after status qualification')
        targets.append(dict(entry=entry,note_path=note['path']))
    owner.require((owner.PROJECT/'.tuldok/project.json').read_bytes() == raw,'Project changed during correction before-custody')
    owner.JOB.mkdir()
    owner.save(owner.BEFORE,dict(status='compact_pressure_status_note_before_custody_captured',created_unix=time.time(),
        project_id=snapshot['project_id'],checkpoint=snapshot['checkpoint'],header_sha256=owner.digest(raw),
        canonical_records=len(snapshot['inventory']),old_target_entries=targets,payloads=owner.payloads(),
        preparation=owner.binding(owner.PREPARED/'preparation.json'),previous_note_custody=owner.binding(OLD_CUSTODY),
        no_full_header_project_or_payload_backup=True,owner_configuration=owner.binding(Path(__file__))))
    print('Compact26status-note before-custody captured;no working notes changed.',flush=True)


def after():
    owner.after()
    proof = owner.read(owner.REPORT)
    proof.update(owner_configuration=owner.binding(Path(__file__)),current_reproduction_status_corrected=True,
        all23five_scientific_summary_fields_unchanged=True,overview_words_unchanged=True,
        previous_note_custody=owner.binding(OLD_CUSTODY),measurement_sources_and_payloads_unchanged=True,
        guide_revision=9)
    owner.save(owner.REPORT,proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('before','adopt','after'))
    args = parser.parse_args()
    {'before':before,'adopt':owner.adopt,'after':after}[args.phase]()
