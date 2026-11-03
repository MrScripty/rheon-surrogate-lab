"""Preregister the paired timing pilot, then the22remaining original fits."""
import argparse
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require
from original_train_pressure_accuracy_02 import publication_gate

HERE = Path(__file__).resolve().parent
PILOT_FIT = 'fit-8992e19d389ea016'


def run(mode):
    publication_gate()
    path = RUN/f'original-pressure-GPU-timing-{mode}-protocol-01.json'
    output = RUN/f'original-pressure-GPU-timing-{mode}-01'
    require(not path.exists() and not output.exists(),'Fresh timing protocol and output required')
    accuracy_path = DOC/'reports/original-train-pressure-CUDA-accuracy-01.json'
    accuracy = read(accuracy_path)
    require(accuracy['status'] == 'all23_original_training_pressure_fits_GPU_accuracy_independently_reduced'
            and accuracy['fits'] == 23 and accuracy['qualified_unique_pressure_requests'] == 2944,
            'All23independently reduced original accuracy results required')
    baseline_path = RUN/'pressure-portfolio-GPU-timing-protocol-02.json'
    baseline = read(baseline_path)
    prior = read(RUN/'remaining-original-train-pressure-CUDA-protocol-01.json')
    material = read(ROOT/prior['material']['path'])
    all_ids = [row['fit_item_id'] for row in accuracy['rows']]
    require(set(all_ids) == {row['fit_item_id'] for row in material['models']} and len(all_ids) == 23,
            'Complete original23fit identities required')
    accepted_pilot = None
    if mode == 'batch':
        accepted_pilot = binding(DOC/'reports/original-pressure-GPU-timing-pilot-01.json')
        pilot = read(ROOT/accepted_pilot['path'])
        require(pilot['status'] == 'original_pressure_GPU_timing_pilot_independently_reduced'
                and pilot['fits'] == 1 and pilot['rows'][0]['fit_item_id'] == PILOT_FIT
                and pilot['fresh_processes'] == 2,'Exact paired CUDA pilot required before expansion')
    fit_ids = [PILOT_FIT] if mode == 'pilot' else [name for name in reversed(all_ids) if name != PILOT_FIT]
    sources = dict(prior['source_sha256'])
    for source in HERE.glob('*.py'):
        spec = binding(source)
        sources[spec['path']] = spec['sha256']
    selected_accuracy = {row['fit_item_id']:row['result'] for row in accuracy['rows'] if row['fit_item_id'] in fit_ids}
    for spec in [binding(accuracy_path),baseline['frozen_cases'],binding(baseline_path),
                 binding(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'),*selected_accuracy.values()]:
        require(binding(ROOT/spec['path']) == spec,'Selected timing evidence changed')
        sources[spec['path']] = spec['sha256']
    # Two phase blocks reverse fit order; each worker also reverses case order.
    blocks = [dict(fit_item_id=fit_id,phase=phase) for phase in (0,1)
              for fit_id in (fit_ids if phase == 0 else list(reversed(fit_ids)))]
    value = dict(schema='rheon.original-train-pressure-GPU-timing.v1',created_unix=time.time(),mode=mode,device='cuda',
        expected_GPU=prior['expected_GPU'],fit_ids=fit_ids,block_order=blocks,source_sha256=sources,
        output_root=str(output.relative_to(ROOT)),worker=str((HERE/'original_train_pressure_timing.py').relative_to(ROOT)),
        batch_owner=str((HERE/'run_original_pressure_timing.py').relative_to(ROOT)),
        material=prior['material'],accuracy_reduction=binding(accuracy_path),accuracy=selected_accuracy,
        runtime_manifest=binding(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'),
        frozen_cases=baseline['frozen_cases'],publication_acceptance=prior['publication_acceptance'],
        baseline_timing_protocol=binding(baseline_path),accepted_pilot=accepted_pilot,
        host_cpu=baseline['host_cpu'],warmups_per_case=baseline['warmups_per_case'],samples_per_case=baseline['samples_per_case'],
        calibration_pairs_per_case=4,cases_per_fit=128,worker_deadline_seconds=1200,
        aggregate_deadline_seconds=3600 if mode == 'pilot' else 14400,minimum_free_disk_bytes=512*1024**2,
        maximum_Torch_GPU_allocated_bytes=baseline['maximum_Torch_GPU_allocated_bytes'],
        maximum_worker_RSS_bytes=baseline['maximum_worker_RSS_bytes'],timing_scope=baseline['timing_scope'],
        primary_observer_hooks=False,calibration='4paired on/off calls per case; order alternates by phase,case,pair. No overhead subtraction or zero clipping.',
        resource_scope='Before/every32cases/after GPU compute snapshots; not continuous exclusivity. Host load and GPU clocks/temperature retained.',
        baseline_comparison='Same128consumed old-solver pressure cases,10primary samples/phase,5warmups,CPU4,1Torchthread and complete CPU/CUDA/CPU boundary as127named fits. Separate paired observer calibration is new. Campaigns are not simultaneous paired optimization trials.',
        new_parameter_updates=0,new_reference_generation=False,full_simulation_step=False,independent_holdout=False,
        own_history_drift=False,full_portfolio_complete=False,producer=binding(Path(__file__)))
    save(path,value)
    print(mode,len(fit_ids),'fits',len(blocks),'fresh GPU process phases preregistered',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pilot','batch'),required=True)
    args = parser.parse_args()
    run(args.mode)
