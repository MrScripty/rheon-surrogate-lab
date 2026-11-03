"""Route the 35 nonportable states to real fit outputs or explicit other roles.

Hashes existing bytes in place. Checks original receipt/evidence claims, records
current artifact-owner constraints, and copies/deletes no payload. Static owner
inspection is not a native import or a qualification of a new retention API.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1/reports'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
SOURCE = RUN/'sources/Tuldok-composition-source-artifacts'
OUTPUT = DOC/'remaining-parameter-retention-routing-02.json'


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path.relative_to(ROOT)),sha256=digest)


def require(value,message):
    if not value:
        raise ValueError(message)


def audit():
    require(not OUTPUT.exists(),'Preserve the original routing audit')
    paths = [DOC/'nonportable-checkpoint-availability-01.json',RUN/'discovery/checkpoint-contexts-01.json',
             RUN/'discovery/retained-model-locations-01.json',RUN/'discovery/model-worklist-05.json']
    missing, contexts, locations, work = [read(p) for p in paths]
    contexts = {r['sha256']:r for r in contexts['rows']}
    locations = {r['sha256']:r for r in locations['checkpoints']}
    fits = {r['item_id']:r for r in work['fit_cards']}
    header_path = PROJECT/'.tuldok/project.json'
    header_raw = header_path.read_bytes();header = json.loads(header_raw)
    declared = {e['id'] for e in header['inventory'] if e['family'] == 'project_artifact'}
    require(len(locations) == 524 and sum(identity in declared for identity in locations) == 489,
            'Complete known-state inventory or current native declaration count differs')
    rows, receipt_cache, evidence_cache = [], {}, {}
    for state in missing['states']:
        identity = state['sha256'];context = contexts[identity]
        require(identity not in declared and not (PROJECT/'artifacts/sha256'/identity).exists(),
                'A previously nonportable state is already retained; reconcile current membership')
        candidates = [ROOT/state['retained_path'],ROOT/locations[identity]['retained_path']]
        path = next((p for p in candidates if p.is_file()),None)
        require(path is not None and path.stat().st_size == state['bytes']
                and binding(path)['sha256'] == identity,'Retained parameter bytes unavailable or changed')
        fit_bindings = []
        for output in context.get('fit_outputs',[]):
            fit_id = output['fit_item_id']
            if fit_id not in receipt_cache:
                fit = fits[fit_id]
                candidates = [ROOT/p for p in fit['receipt_paths']]
                selected = next((p for p in candidates if p.is_file()
                    and binding(p)['sha256'] == fit['receipt_sha256']),None)
                require(selected is not None,'Original fit receipt unavailable')
                receipt_cache[fit_id] = (selected,read(selected))
            receipt_path,receipt = receipt_cache[fit_id]
            claim = receipt['outputs'][output['name']]
            require(claim['sha256'] == identity and claim['bytes'] == state['bytes'],
                    'State is not the original recorded fit output')
            fit_bindings.append(dict(**output,receipt=binding(receipt_path)))
        evidence = []
        for spec in context.get('evidence',[]):
            original = ROOT/spec['path']
            if spec['path'] not in evidence_cache:
                # Existing compaction remaps are authoritative locators, never
                # permission to replace an original source identity.
                if not original.is_file():
                    remaps = read(paths[2]).get('evidence_remaps',[])
                    matches = [r for r in remaps if r.get('previous_path') == spec['path']
                               or r.get('path') == spec['path'] or r.get('original_path') == spec['path']]
                    require(len(matches) == 1,'Retained evidence needs an explicit original-path remap')
                    remap = matches[0]
                    original = ROOT/remap['retained_path']
                require(binding(original)['sha256'] == spec['sha256'],'Original contextual evidence changed')
                evidence_cache[spec['path']] = original
            actual = evidence_cache[spec['path']]
            evidence.append(dict(original_path=spec['path'],**binding(actual)))
        if context.get('weight_claim'):
            source_index,keys = context['weight_claim']
            value = read(ROOT/evidence[source_index]['path'])
            for key in keys:
                value = value[key]
            require(value == identity,'Contextual weight claim differs from actual bytes')
        require(bool(fit_bindings) == (state['role'] == 'recorded_train_receipt_output'),
                'Unrecorded assembly/control/failed-state context was represented as a successful fit')
        rows.append(dict(sha256=identity,bytes=state['bytes'],retained_path=str(path.relative_to(ROOT)),
            role=state['role'],original_fit_outputs=fit_bindings,context_evidence=evidence,
            interpretation=context.get('interpretation'),native_artifact_declared=False,native_payload_present=False,
            disposition='Keep exact existing bytes; native retention and model-appropriate benchmark/reproduction remain required.'))
    require(len(rows) == len({r['sha256'] for r in rows}) == 35
            and len(receipt_cache) == 2 and sum(r['bytes'] for r in rows) == 41115369
            and sum(r['role'] == 'recorded_train_receipt_output' for r in rows) == 21,
            'Complete original 35-state routing required')
    source_files = [SOURCE/name for name in ('project_sources.py','project_material.py','project_projection.py','agent_api.py')]
    texts = {p.name:p.read_text() for p in source_files}
    require("record=self.materials.capture_snapshot(saved,raw,catalog_id=origin)" in texts['project_sources.py']
            and "artifacts=_edges([item],'project_snapshot')" in texts['project_material.py']
            and "if declaration is None:" in texts['project_projection.py']
            and "Artifact is not declared by this project." in texts['project_projection.py'],
            'Inspected retention-owner constraints changed')
    require(header_path.read_bytes() == header_raw,'Native checkpoint changed during this read-only audit')
    value = dict(status='all35_nonportable_states_routed_to_original_evidence_and_retention_gap',created_unix=time.time(),
        project_checkpoint=header['checkpoint'],known_parameter_states=524,native_declared_states=489,
        outside_states=35,outside_bytes=41115369,role_counts=dict(Counter(r['role'] for r in rows)),
        recorded_successful_fit_output_states=21,original_successful_fit_identities=sorted(receipt_cache),
        other_context_states=14,all_retained_weight_bytes_freshly_hashed=True,all_available_original_receipt_and_context_claims_checked=True,
        static_owner_finding='Source registration retains the manifest artifact, not all data-reference bytes. Artifact resolution requires an existing project_artifact declaration. These operations alone do not import the 35 undeclared states.',
        required_followup='Qualify an explicit native retention path for source-bound opaque artifacts and context records, or prove an existing semantically correct owner path. Do not fabricate Runs, Trials, material admissions or scientific field data to retain weights.',
        limitation='Source inspection, receipt/context routing and actual byte availability only. No new native API, import, numerical equivalence, clean retraining or complete model coverage is qualified.',
        rows=rows,new_neural_forwards=0,native_mutations=0,payload_copies=0,payload_deletions=0,
        full_portfolio_complete=False,inputs=[binding(p) for p in paths],source_bindings=[binding(p) for p in source_files],
        producer=binding(Path(__file__)))
    OUTPUT.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    print('35 states routed: 21 outputs of two original fits; 14 assembly/control/legacy/failed contexts. No payload copied or removed.')


if __name__ == '__main__':
    audit()
