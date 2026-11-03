"""Advance exact default-fit coverage after the23pressure timings reduce."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,DOC,PROJECT,read,save,binding,require
from original_train_pressure_accuracy_02 import publication_gate


def unchanged_fit_facts(before,after):
    """Only the already-published generator card links/outcomes may differ."""
    require({k:v for k,v in before.items() if k not in ('fits','sources')} ==
            {k:v for k,v in after.items() if k not in ('fits','sources')},'Index identity/aliases changed')
    changed = 0
    for old,new in zip(before['fits'],after['fits'],strict=True):
        require({k:v for k,v in old.items() if k not in ('card_sha256','matched_GPU_benchmark')} ==
                {k:v for k,v in new.items() if k not in ('card_sha256','matched_GPU_benchmark')},
                'Original fit/checkpoint facts changed')
        if old != new:
            require(old['task'] == 'full_state_generator' and 'matched_GPU_benchmark' not in old
                    and 'matched_GPU_benchmark' in new,'Unexpected fit outcome revision')
            changed += 1
    require({k:v for k,v in after['sources'].items() if k != 'matched_generator_GPU_benchmark'} == before['sources']
            and 'matched_generator_GPU_benchmark' in after['sources'],'Unexpected source revision')
    return changed


def current_inventory(previous):
    publication_gate()
    from import_common_generator_measurements import RUN,header,digest
    _,snapshot = header()
    proposal = RUN/'native-matched-generator-outcome-notes-01/preparation.json'
    approved = next(row for row in read(proposal)['notes'] if row['path'] == 'notes/portfolio-index.json')
    current_path = PROJECT/'notes/portfolio-index.json'
    current_binding = binding(current_path)
    require(approved['previous_sha256'] == previous['source_fit_index']['sha256']
            and approved['sha256'] == current_binding['sha256']
            and binding(ROOT/approved['prepared_path'])['sha256'] == current_binding['sha256'],
            'Only the accepted exact matched-outcome index transition is supported')
    candidates = []
    for entry in snapshot['inventory']:
        if entry['family'] != 'project_file' or entry['id'] != approved['file_id']:
            continue
        path = PROJECT/'.tuldok'/entry['path']
        raw = path.read_bytes()
        require(len(raw) == entry['bytes'] and digest(raw) == entry['sha256'],'Canonical index record changed')
        document = json.loads(raw)
        if document['project_id'] == snapshot['project_id']:
            candidates.append((path,document))
    require(len(candidates) == 1,'Exactly one current-owner canonical index record required')
    path,document = candidates[0]
    require(document['current']['sha256'] == current_binding['sha256']
            and document['current']['text'].encode() == current_path.read_bytes(),'Local and canonical index differ')
    versions = [row for row in document['history'] if row['revision'] == approved['expected_revision']
                and row['sha256'] == approved['previous_sha256']]
    require(len(versions) == 1 and hashlib.sha256(versions[0]['text'].encode()).hexdigest() == approved['previous_sha256'],
            'Prior exact coverage index must remain in canonical native history')
    changed = unchanged_fit_facts(json.loads(versions[0]['text']),read(current_path))
    require(changed == 35,'Only35published generator outcomes may differ; all205fit/checkpoint facts remain exact')
    return current_binding,dict(previous_index=previous['source_fit_index'],current_index=current_binding,
        native_record=binding(path),approved_proposal=binding(proposal),changed_generator_outcomes=35,
        original205fit_checkpoint_and_alias_facts_exact=True)


def run():
    output = DOC/'reports/common-fit-benchmark-coverage-03.json'
    require(not output.exists(),'Preserve original coverage evidence')
    previous_path = DOC/'reports/common-fit-benchmark-coverage-02.json'
    previous = read(previous_path)
    timing_path = DOC/'reports/original-pressure-GPU-timing-batch-01.json'
    timing = read(timing_path)
    require(timing['status'] == 'original_pressure_GPU_timing_batch_independently_reduced' and timing['fits'] == 23
            and timing['fresh_processes'] == 46 and timing['all_output_hashes_exact'],
            'Complete independently reduced23fit timings required')
    index_binding,index_transition = current_inventory(previous)
    rows = copy.deepcopy(previous['rows'])
    fits = {row['fit_item_id']:row for row in rows}
    for measured in timing['rows']:
        row = fits[measured['fit_item_id']]
        require(row['status'] == 'common_GPU_accuracy_available_latency_pending' and row['default_benchmark'] is None
                and row['accuracy_benchmark']['result'] == measured['accuracy']
                and row['accuracy_benchmark']['worst'] == measured['worst'],'Timing must join exact prior accuracy-only fit')
        row['default_benchmark'] = dict(group='pressure_request',identifier=measured['fit_item_id'],cases=128,
            source='original_training_output_CUDA_restore',benchmark_authority=binding(timing_path),
            accuracy_authority=row['accuracy_benchmark']['independent_reduction'],
            timing_units='GPU ms/pressure request; not a simulation step',
            case_median_median_ms=measured['case_median_median_ms'])
        row['accuracy_benchmark']['latency_qualified'] = True
        row['status'] = 'default_common_GPU_accuracy_and_timing_available'
    pending = [row for row in rows if row['default_benchmark'] is None]
    pressure = [row for row in rows if row['default_benchmark'] and row['default_benchmark']['group'] == 'pressure_request']
    require(len(rows) == len(fits) == 205 and len(pressure) == 150 and len(pending) == 20
            and sum(row['status'] == 'default_common_GPU_accuracy_and_timing_available' for row in rows) == 185,
            'Exact150pressure+35generator default fits and20pending components required')
    report = dict(previous,status='all205_fit_identities_joined_to185_default_common_benchmarks_and20_pending',
        created_unix=time.time(),previous_coverage=binding(previous_path),new_timing=binding(timing_path),rows=rows,
        default_common_benchmark_fits=185,pressure_request_fits=150,common_benchmark_pending_fits=20,
        additional_pressure_latency_pending_fits=0,pending_fit_ids=[row['fit_item_id'] for row in pending],
        pending_by_task=dict(sorted(Counter(row['task'] for row in pending).items())),producer=binding(Path(__file__)),
        scope='Exact default-fit GPU accuracy/timing coverage:150pressure-only fits on128requests and35complete generators on64cases. Distinct tasks. Not complete524parameter states,intended variants,independent holdout,long drift or clean training reproduction.',
        source_fit_index=index_binding,index_revision_reconciliation=index_transition,
        evidence=[*previous['evidence'],binding(timing_path)],new_model_forwards=0,native_mutations=0,full_portfolio_complete=False)
    save(output,report)
    print('185/205default fits now have common GPU accuracy/timing:150pressure,35generators;20components remain.',flush=True)


if __name__ == '__main__':
    run()
