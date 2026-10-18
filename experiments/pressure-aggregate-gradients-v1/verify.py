"""Check retained source/vector identities, projection arithmetic and limitations."""
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def verify(directory, output):
    result = read(directory/'results.json')
    assert result['status'] == 'completed'
    for name, identity in result['sources'].items():
        assert sha(ROOT/name) == identity, name
    checks, limited = [], 0
    for identifier, projection in result['models'].items():
        row = read(directory/'reports'/(identifier+'-gradients.json'))
        for name, identity in {**row['sources'], **row['vectors']}.items():
            assert sha(ROOT/name) == identity, name
        assert len(row['vectors']) == 40
        assert sum((ROOT/name).stat().st_size for name in row['vectors']) <= 64*1024**2
        assert row['vector_workspace_bound_bytes'] <= 16*1024**2 and row['fitting_tensor_bytes'] <= 8*1024**2
        assert row['optimizer_updates'] == row['teacher_forwards'] == 0
        assert row['inventory_unchanged'] and row['fitted_pressure_replay_exact']
        assert row['threads'] == 1 and not row['cuda_initialized']
        assert row['optimizer_direction']['status'] == 'unavailable'
        if identifier in ('C53','C66'):
            assert row['prior_replay']['exact_component_geometry_and_full_losses']
        assert projection['policies'] == row['policies'] and projection['cases'] == row['cases']
        for case in row['cases']:
            assert case['joint_gradient_exact'] and case['operator_defect'] < 1e-10
            assert math.isclose(sum(case['values'].values()),case['full_value'],rel_tol=1e-12,abs_tol=1e-14)
            assert case['separate_gradient_precision_check_passed'] == (case['separate_gradient_relative_l2'] <= 5e-6)
        for policy in row['policies'].values():
            assert policy['direct_vector_projection_check']
            assert math.isclose(sum(case['signed_mean_contribution'] for case in policy['cases']),1.,abs_tol=1e-12)
            for case in policy['cases']:
                expected = sum(other['pre_mean_cap_scale']*row['across']['full']['dot_matrix'][case['case']][other['case']] for other in policy['cases'])/8
                assert math.isclose(case['dot_mean'], expected, rel_tol=1e-12, abs_tol=1e-12)
                for term, value in case['terms'].items():
                    assert math.isfinite(value['dot_mean'])
                    assert math.isclose(value['derivative_along_negative_clipped_mean'], -policy['global_clip_scale']*value['dot_mean'], rel_tol=1e-12, abs_tol=1e-12)
                    norm = row['across'][term]['norms'][case['case']]
                    if norm and policy['mean_l2']:
                        assert math.isclose(value['cosine_mean'], value['dot_mean']/norm/policy['mean_l2'], rel_tol=1e-12, abs_tol=1e-12)
        count = sum(not case['separate_gradient_precision_check_passed'] for case in row['cases'])
        limited += count
        checks.append(dict(model=identifier, cases=8, unchanged_inventory=True,
            separate_gradient_precision_limited_cases=count, actual_adam_direction='unavailable'))
    assert limited == result['decision']['separate_gradient_precision_limited_cases']
    assert result['decision']['joint_gradient_equivalence_passes'] == 24
    assert not result['decision']['training_started'] and not result['decision']['model_promoted']
    execution = result['execution']
    config = read(ROOT/'experiments/pressure-aggregate-gradients-v1/config.json')
    assert execution['runtime_manifest_sha256'] == sha(ROOT/config['runtime'])
    assert execution['source_sha256'] == sha(Path(__file__).with_name('diagnose.py'))
    assert execution['status'] == 'completed' and len({row['cpu'] for row in execution['workers']}) == 3
    assert all(row['reaped'] and row['returncode'] == 0 for row in execution['workers'])
    math_proof = read(directory/'reports/math-check.json')
    joint = read(directory/'reports/joint-guard.json')
    cancel = read(directory/'reports/cancellation-final.json')
    assert math_proof['status'] == joint['status'] == cancel['status'] == 'passed'
    assert math_proof['aggregation_source_sha256'] == sha(Path(__file__).with_name('aggregation.py'))
    assert joint['measure_source_sha256'] == sha(Path(__file__).with_name('measure.py'))
    assert cancel['supervisor_source_sha256'] == sha(Path(__file__).with_name('diagnose.py'))
    assert cancel['all_children_reaped'] and cancel['parent_reaped'] and cancel['partial_results_not_published']
    write(output, dict(status='passed_with_retained_precision_limits', checks=checks,
        joint_equivalence_cases=24, separate_gradient_precision_limited_cases=limited,
        source_sha256=sha(__file__), results_sha256=sha(directory/'results.json'), all_owned_workers_reaped=True,
        scope='Source/vector identity, literal math, direct-vector projection, captured-joint guard and no-update diagnostics; not causal training, actual Adam or pressure improvement.'))
    print('Verified aggregate diagnostic; retained',limited,'separate-gradient precision limits',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.directory.resolve(), args.output.resolve())
