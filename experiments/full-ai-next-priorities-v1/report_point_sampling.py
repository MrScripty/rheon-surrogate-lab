"""Publish learned point-component custody and native checks, separate from forecasts."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'point-sampling-results.json'
    if output.exists():raise ValueError('Fresh learned point report required')
    pair_path=BASE/'point-sampling-pair-01';pair=read(pair_path/'receipt.json')
    train=read(pair_path/'train/receipt.json');restore=read(pair_path/'restore/receipt.json')
    history=read(pair_path/'restore/history.json')
    evaluation_path=BASE/'point-sampling-native-evaluation-01.json';evaluation=read(evaluation_path)
    if pair['status']!='completed'or pair['optimizer_updates']!=1000 or not all(pair['artifact_parity'].values()):
        raise ValueError('Complete managed component pair required')
    if restore['closure']!=train['closure']or restore['optimizer_updates']!=0 or restore['execution']['cuda_initialized']:
        raise ValueError('Exact zero-update CPU custody required')
    if restore['originating_training_run']['run_id']!=pair['training_run_id']or not restore['full_cpu_probe_bitwise_verified']:
        raise ValueError('CPU restore provenance differs')
    if not history['coefficient_accuracy_pass']or evaluation['status']!='passed_component_qualification'or len(evaluation['rows'])!=24:
        raise ValueError('Registered component accuracy checks failed')
    for name,digest in evaluation['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Point qualification evidence changed')
    registration_path=DOCS/'point-sampling-registration-01.json';registration=read(registration_path)
    for name,digest in registration['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered point source changed')
    for phase,receipt in (('train',train),('restore',restore)):
        terminal=read(pair_path/f'terminal-{pair["training_run_id"if phase=="train"else"restore_run_id"]}.json')
        if terminal['state']!='completed'or terminal['custody_phase']!='drained':raise ValueError('Terminal drained managed Run required')
        for name,spec in receipt['outputs'].items():
            if sha(pair_path/phase/name)!=spec['sha256']:raise ValueError('Published component artifact changed')
    first=read(BASE/'point-sampling-batch-01/execution.json');recovery=read(BASE/'point-sampling-evaluation-batch-02/execution.json')
    if first['status']!='stopped'or not first['owned_child_reaped']or recovery['status']!='completed'or not recovery['owned_child_reaped']:
        raise ValueError('Retained first failure and complete recovered evaluation required')
    paths=[registration_path,DOCS/'point-sampling-evaluation-protocol-01.json',DOCS/'point-sampling-evaluation-recovery-01.json',
        pair_path/'receipt.json',pair_path/'train/receipt.json',pair_path/'restore/receipt.json',pair_path/'restore/history.json',
        pair_path/'restore/weights.npz',evaluation_path,BASE/'point-sampling-batch-01/execution.json',
        BASE/'point-sampling-evaluation-batch-02/execution.json',BASE/'point-sampling-cpu-qualification-01.json',BASE/'point-sampling-cuda-qualification-01.json']
    write(output,dict(status='completed_component_calibration',parameters=972,optimizer_updates=1000,
        training_run_id=pair['training_run_id'],restore_run_id=pair['restore_run_id'],exact_CPU_artifact_parity=pair['artifact_parity'],
        worst_synthetic_development_error_float64=history['worst_development_relative_error_float64'],
        worst_synthetic_development_error_float32=history['worst_development_relative_error_published_float32'],
        worst_native_geometry_error_float32=evaluation['maximum_native_relative_error'],native_cases=24,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in paths},reporter_sha256=sha(__file__),
        full_state_forecast=False,CPU_latency_claim=False,candidate_frontier_point=False,reserved_scenes_consumed=False,
        original_goal_complete=False,next_gate='Qualify the actual causal full-state displacement model and training objective before a bounded managed fit'))
    (DOCS/'point-sampling-results.md').write_text(
        '# Learned point-sampling component\n\n'
        'The 972-coefficient layer completes 1,000 ordinary managed CUDA SGD updates from zero. '
        'Its fresh CPU restore matches all five saved artifacts byte for byte. '
        'The worst error on 16 held-out synthetic patterns is '
        f'{history["worst_development_relative_error_float64"]:.4g} in float64 and '
        f'{history["worst_development_relative_error_published_float32"]:.4g} after float32 publication.\n\n'
        'The restored float32 layer passes all 24 independent native point checks across '
        'four domain shapes, both spatial densities and zero/small/unit random fields. '
        f'Worst native relative error is {evaluation["maximum_native_relative_error"]:.4g}; zero input remains exact.\n\n'
        'No analytical sampling coefficients were assigned. This layer predicts negative '
        'point velocities through a learned convolution. It is an initialization component '
        'for the displacement model, with no complete-state accuracy, CPU latency or rollout claim.\n\n'
        'The first evaluation reached a reporting-path error after native comparisons. '
        'Its failed batch and log remain preserved. Supplying an absolute input path completed '
        'the same evaluator and protocol; no weights changed and no fit was resubmitted. '
        'Both batch controllers reaped their children.\n\n'
        '[Evidence and source hashes](point-sampling-results.json) · '
        '[Calibration registration](point-sampling-registration-01.json) · '
        '[Full-state representation qualification](displacement-transport-results.md)\n')
    print('Published managed learned-point component; full-state goals remain open',flush=True)


if __name__=='__main__':run()
