"""Publish qualified offline transport labels without a model-performance claim."""
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


def run():
    output = DOCS / 'displacement-transport-results.json'
    if output.exists():
        raise ValueError('Fresh displacement transport report required')
    protocol_path = DOCS / 'displacement-transport-qualification-protocol-01.json'
    protocol = read(protocol_path)
    build = BASE / 'transport-teacher-build-01/build-receipt.json'
    evidence = {str(p.relative_to(ROOT)): sha(p) for p in (protocol_path, build)}
    summaries = []
    for device in ('cpu', 'cuda'):
        path = BASE / f'displacement-transport-{device}-qualification-01.json'
        data = read(path)
        assert data['status'] == 'passed' and data['device'] == device
        assert data['protocol_sha256'] == sha(protocol_path)
        assert data['native_build_receipt_sha256'] == sha(build)
        assert data['source_sha256'] == protocol['source_sha256']
        for name, digest in data['source_sha256'].items():
            assert sha(HERE / name) == digest, name
        for name, digest in data['source_evidence'].items():
            assert sha(ROOT / name) == digest, name
        rows = data['rows']
        assert len(rows) == protocol['cases_per_device']
        assert sum(r['kind'] == 'captured' for r in rows) == protocol['captured_training_cases']
        assert sum(r['kind'] == 'fixture' for r in rows) == protocol['manufactured_cases']
        assert all(r['deterministic_field_and_shift_gradients'] and r['zero_and_constant_tracer_exact'] for r in rows)
        limits = {
            'maximum_displacement_absolute_error_cells': 'maximum_displacement_absolute_error_cells',
            'maximum_point_velocity_relative_error': 'point_velocity_relative_tolerance',
            'maximum_native_readout_relative_error': 'native_readout_relative_tolerance',
        }
        for name, limit in limits.items():
            assert data[name] <= protocol[limit]
        assert data['optimizer_updates'] == data['native_pressure_solves'] == data['native_steps_executed'] == 0
        assert not data['candidate_predictor_implemented'] and not data['candidate_frontier_point']
        assert not data['reserved_scenes_consumed'] and data['teacher_labels_for_training_only']
        evidence[str(path.relative_to(ROOT))] = sha(path)
        summaries.append(dict(device=device, cases=len(rows), **{n: data[n] for n in limits}))
    write(output, dict(status='completed_offline_qualification', devices=summaries,
        source_evidence=evidence, reporter_sha256=sha(__file__), optimizer_updates=0,
        candidate_predictor_implemented=False, candidate_frontier_point=False,
        CPU_latency_claim=False, full_state_accuracy_claim=False, reserved_scenes_consumed=False))
    (DOCS / 'displacement-transport-results.md').write_text(
        '# Face and cell transport representation qualification\n\n'
        'CPU and CUDA each pass all 80 registered cases: 64 captured training windows '
        'and 16 manufactured fields, covering four domain shapes and both spatial densities. '
        'The training-only tensor labels match native face/cell displacement and point-velocity '
        'labels exactly. Given native displacements, the fixed readout reproduces native '
        'velocity and tracer outputs exactly at float32 precision.\n\n'
        'Gradients with respect to fields and displacements are finite and bitwise '
        'repeatable on each device. Zero and constant tracer behavior also passes. '
        'The captured next-tracer fields match the native primitive bitwise.\n\n'
        'These checks qualify labels and a representation, with zero optimizer updates. '
        'They do not establish a learned forecast, full-state rollout accuracy or CPU latency. '
        'A candidate must predict its own displacements from causal inputs; native '
        'backtracing and reference fields remain training-only. Every requested horizon '
        'and the complete 4/6/8 ms block budgets still apply.\n\n'
        '[Evidence and source hashes](displacement-transport-results.json) · '
        '[Frozen protocol](displacement-transport-qualification-protocol-01.json) · '
        '[Timing and architecture definitions](multi-step-generation-options.md)\n')
    print('Published CPU/CUDA transport qualification; no candidate accuracy or timing claim')


if __name__ == '__main__':
    run()
