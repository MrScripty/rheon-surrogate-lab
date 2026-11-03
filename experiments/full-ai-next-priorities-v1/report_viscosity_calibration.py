"""Publish managed neural viscosity learning without a full-state success claim."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'viscosity-calibration-results.json'
    if output.exists():raise ValueError('Fresh viscosity component report required')
    pair_path=BASE/'viscosity-calibration-pair-01';pair=read(pair_path/'receipt.json')
    trained=read(pair_path/'train/receipt.json');restored=read(pair_path/'restore/receipt.json');history=read(pair_path/'restore/history.json')
    evaluation_path=BASE/'viscosity-calibration-native-evaluation-01.json';evaluation=read(evaluation_path)
    if pair['status']!='completed'or pair['optimizer_updates']!=1000 or not all(pair['artifact_parity'].values()):
        raise ValueError('Exact complete managed viscosity fit/restore required')
    if trained['closure']!=restored['closure']or restored['optimizer_updates']!=0 or not restored['full_cpu_probe_bitwise_verified']or restored['execution']['cuda_initialized']:
        raise ValueError('Closed zero-update CPU restoration required')
    if evaluation['status']!='passed_component_qualification'or len(evaluation['rows'])!=24 or not history['coefficient_accuracy_pass']:
        raise ValueError('Synthetic and native component checks must pass')
    reg_path=DOCS/'viscosity-calibration-registration-01.json';reg=read(reg_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered calibration source changed')
    for name,digest in evaluation['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Native component evidence changed')
    for phase,receipt in (('train',trained),('restore',restored)):
        for name,spec in receipt['outputs'].items():
            if sha(pair_path/phase/name)!=spec['sha256']:raise ValueError('Published component artifacts changed')
    paths=[pair_path/'receipt.json',pair_path/'train/receipt.json',pair_path/'restore/receipt.json',pair_path/'restore/history.json',
        pair_path/'restore/weights.npz',evaluation_path,reg_path,BASE/'viscosity-calibration-teacher-01/manifest.json',
        BASE/'viscosity-calibration-cpu-qualification-01.json',BASE/'viscosity-calibration-cuda-qualification-01.json',
        BASE/'viscosity-calibration-batch-01/execution.json']
    write(output,dict(status='completed_managed_neural_component_calibration',parameters=243,optimizer_updates=1000,
        training_run_id=pair['training_run_id'],restore_run_id=pair['restore_run_id'],exact_CPU_artifact_parity=pair['artifact_parity'],
        worst_synthetic_development_error_float64=history['worst_development_relative_error_float64'],
        worst_synthetic_development_error_float32=history['worst_development_relative_error_published_float32'],
        worst_native_operator_error_float32=evaluation['maximum_operator_relative_error'],
        worst_native_physical_update_error=evaluation['maximum_updated_velocity_relative_error'],native_cases=24,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},reporter_sha256=sha(__file__),
        numerical_viscosity_in_candidate_inference=False,analytic_coefficient_assignment=False,
        full_state_accuracy_claim=False,CPU_latency_claim=False,reserved_scenes_consumed=False))
    (DOCS/'viscosity-calibration-results.md').write_text(
        '# Managed neural viscosity calibration\n\n'
        'The 243-coefficient neural layer completes 1,000 managed CUDA SGD updates '
        'from zero using exact native strain-update labels. The fresh CPU restore '
        'matches all five saved artifacts byte for byte.\n\n'
        f'Worst held-out synthetic error is {history["worst_development_relative_error_float64"]:.4g} '
        f'in float64 and {history["worst_development_relative_error_published_float32"]:.4g} after float32 publication. '
        f'All 24 native geometry/density checks pass: worst operator error is {evaluation["maximum_operator_relative_error"]:.4g}; '
        f'worst physical-update error is {evaluation["maximum_updated_velocity_relative_error"]:.4g}.\n\n'
        'No analytical stencil coefficients were assigned. Numerical viscosity remains '
        'a training/evaluation teacher; candidate inference uses the learned convolution. '
        'These are component results. The composed full-state model must separately '
        'pass every output, own240 and actual CPU timing.\n\n'
        '[Receipts, weights and source evidence](viscosity-calibration-results.json) · '
        '[Frozen registration](viscosity-calibration-registration-01.json)\n')
    print('Published learned viscosity coefficient custody and native checks',flush=True)


if __name__=='__main__':run()
