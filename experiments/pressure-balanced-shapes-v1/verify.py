"""Verify retained scientific gates separately from successful execution."""
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
    training = result['training'][0]
    count = len(training['fundamental_bank']['cases'])
    assert training['updates'] == 200 and training['examples'] == count*200 == 1600
    assert training['optimizer_work']['network_forwards'] == 6400
    assert training['optimizer_work']['forward_cells'] == 7424000
    assert result['control']['updates'] == 1600
    assert training['optimizer_work'] == result['control']['training']['optimizer_work']
    assert training['fundamental_bank']['sampling_order_sha256'] == result['control']['training']['fundamental_bank']['sampling_order_sha256']
    assert all(training['artifact_parity'].values())
    assert all(row['state'] == 'completed' and row['custody_phase'] == 'drained' for row in training['managed_custody'])
    assert result['cpu_execution']['all_owned_children_reaped']
    identifier = training['id']
    for model in ('C53', identifier):
        fitted, real, manufactured = result['fitted'][model], result['snapshots'][model], result['diagnostics'][model]
        assert len(fitted['measurements']) == 32 and len(real['rows']) == 512 and len(manufactured['rows']) == 960
        assert not any(row['cuda_initialized'] for row in (fitted, real, manufactured))
        assert fitted['threads'] == real['torch_threads'] == 1
        assert all(math.isfinite(row['relative_equation_residual']) for row in fitted['measurements']+manufactured['rows'])
        assert all(math.isfinite(row['relative_residual']) for row in real['rows'])
    values = lambda model:{(row['case'],row['passes']):row['relative_equation_residual'] for row in result['fitted'][model]['measurements']}
    baseline, candidate = values('C53'), values(identifier)
    expected = dict(four_pass_all_fitted=all(candidate[i,4] <= 1e-6 for i in range(count)),
        eight_pass_all_fitted=all(candidate[i,8] <= 1e-6 for i in range(count)),
        one_pass_no_case_worse=all(candidate[i,1] <= baseline[i,1] for i in range(count)),
        two_pass_no_case_worse=all(candidate[i,2] <= baseline[i,2] for i in range(count)))
    assert expected == result['decision']['capacity_checks']
    assert result['decision']['capacity_gate_passed'] == all(expected.values())
    assert result['decision']['baseline_retained'] == 'C53' and not result['decision']['model_promoted']
    assert result['historical_baseline_quality_exact_rows'] == 512
    write(output, dict(status='passed', scope='Receipt consistency and correct execution; scientific fitting outcome is separate.',
        candidate=identifier, examples=1600, aggregate_updates=200, control_updates=1600,
        fitted_cases=count, consumed_real_cases=128, manufactured_cases=240,
        scientific_gate_passed=all(expected.values()), original_untouched_holdouts_consumed=0,
        results_sha256=sha(directory/'results.json'), source_sha256=sha(__file__)))
    print('Verified balanced-shape evidence; scientific gate', all(expected.values()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.directory.resolve(), args.output.resolve())
