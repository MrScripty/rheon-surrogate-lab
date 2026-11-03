"""Preregister exact data/source identities and GPU timing work; no execution."""
from pathlib import Path
import time

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = RUN/'joint-generator-GPU-benchmark-protocol-01.json'


def prepare():
    require(not OUTPUT.exists(), 'Preserve original preregistered GPU protocol')
    cases_path = DOC/'reports/joint-generator-case-preflight-01.json'
    cases = read(cases_path)
    require(cases['status'] == '64_retained_full_state_cases_and_original_reference_operator_verified',
            'Actual original-reference case qualification required')
    inputs = set()
    for model in cases['models']:
        inputs.update(x['path'] for x in [model['configuration'],model['receipt'],model['weights'],
                                         model['normalization'],*model['sources']])
    for case in cases['cases']:
        inputs.add(case['reference_receipt']['path'])
        inputs.add(case['current']['path'])
        inputs.update(frame['path'] for frame in case['reference_leads'])
    inputs.update('experiments/model-portfolio-migration-v1/'+name for name in [
        'joint_generator_gpu_benchmark.py','test_joint_generator_gpu_benchmark.py',
        'full_state_gpu_timing.py','review_direction_pressure_fits.py','captured_pressure_review.py',
        'prepare_method_review_revisions.py','review_pressure_budget_coverage_fits.py',
        'review_targeted_supplement_fits.py','review_water_learning_sequence.py'])
    runtime_path = RUN.parent/'dependency-migration-v1/runtime/current-manifest-v2.json'
    inputs.add(str(runtime_path.relative_to(ROOT)))
    blocks = []
    for phase, models in [(0,cases['models']),(1,list(reversed(cases['models'])))]:
        for model in models:
            blocks.append(dict(identifier=model['identifier'],phase=phase,
                result=f'runs/model-portfolio-migration-v1/joint-generator-GPU-benchmark-01/{model["identifier"]}-phase{phase}.json'))
    protocol = dict(status='preregistered_not_executed',created_unix=time.time(),cases=binding(cases_path),
        worker='experiments/model-portfolio-migration-v1/joint_generator_gpu_benchmark.py',
        runtime_manifest=str(runtime_path.relative_to(ROOT)),dependencies=[binding(ROOT/p) for p in sorted(inputs)],
        models=4,cases_per_model=64,process_blocks=blocks,host_cpu=4,warmups=3,samples=10,
        expected_timed_network_calls=5120,expected_timed_physical_outputs=12800,
        expected_all_network_calls_including_first_and_warmups=7168,
        expected_all_physical_outputs_including_first_and_warmups=17920,
        worker_deadline_seconds=600,max_CUDA_allocated_bytes=4*1024**3,
        device='CUDA',CPU_neural_forwards=0,TF32=False,AMP=False,
        case_order='Ascending original scene/start in phase0; reverse in phase1. ModelKascending then descending.',
        timing='Complete native CPU5fields to allKowned CPU5fields, including source packing/known forcing, H2D, joint forward, reconstruction, allocation/D2H/synchronization. Setup, first calls, warmups, hashes/scoring and finite/ownership oracles separate. Report full block ms and block/K.',
        accuracy='Every physical output/field compared with original native reference; pressure mean-free relative L2, component and combined velocity/tracer relative L2, absolute RMS and sealed FV equation residual. Zero reference norm with nonzero error is undefined, not a passing zero.',
        repeatability='All repeated outputs must match their first-call hashes; independently compare phase outputs during final reduction.',
        contention='Refuse dispatch with other GPU compute; record before/after resources. Competing activity invalidates clean timing qualification and must remain visible.',
        target_interpretation='Diagnostic timing/accuracy only; no acceptance or independent-holdout promotion.1.2ms is obsolete; report actual GPUtime without inferring pressure-request/physical-step equivalence.',
        new_reference_generation=False,payload_copies=0,native_mutations=0,full_portfolio_complete=False,
        actual_CUDA_execution_qualified=False,producer=binding(Path(__file__)))
    OUTPUT.write_bytes(canonical(protocol)+b'\n')
    print('Registered4models/8process blocks/5120timed joint calls; GPU execution still pending.')


if __name__ == '__main__':
    prepare()
