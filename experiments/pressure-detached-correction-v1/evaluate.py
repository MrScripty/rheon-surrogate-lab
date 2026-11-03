"""Measure restored fits on fitting, consumed-real and manufactured populations."""
from pathlib import Path
import argparse
import os
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'runs/pressure-detached-correction-v1'
REPORTS = ROOT / 'docs/plans/flow-reasoning-assessment/execution'
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment
from cpu import material


def run_worker(command, log, env, records, timeout=180):
    started = time.perf_counter()
    with log.open('xb') as stream:
        child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if child.wait(timeout=timeout):
                raise RuntimeError('Evaluation failed; inspect ' + str(log))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            records.append(dict(command=list(map(str, command)), pid=child.pid,
                returncode=child.returncode, reaped=child.poll() is not None,
                elapsed_s=time.perf_counter()-started, log=str(log.relative_to(ROOT))))


def evaluate(*, resume=False):
    campaign = read(DIRECTORY / 'campaign.json')
    execution = read(DIRECTORY / 'execution.json')
    if execution['status'] != 'training_completed' or len(execution['completed']) != 9:
        raise ValueError('All nine managed training/restore workflows must complete before isolated evaluation')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    if available < 14 * 1024**3:
        raise MemoryError('12 GiB host reserve plus one 2 GiB worker required')
    output = DIRECTORY / 'cpu-01'
    output.mkdir(exist_ok=resume)
    config = read(ROOT / campaign['rows'][0]['config'])
    config['arms'] = [read(ROOT / row['config'])['arms'][0] for row in campaign['rows']]
    config_path = output / 'evaluation-config.json'
    if resume:
        if read(config_path) != config:
            raise ValueError('Frozen evaluation configuration changed')
    else:
        write(config_path, config)
    frozen = output / 'material.json'
    if not resume:
        material(config_path, DIRECTORY, frozen, True)
    diagnostic = read(ROOT / 'experiments/pressure-low-mode-tail-v1/config.json')
    diagnostic['baseline_material'] = str(frozen.relative_to(ROOT))
    diagnostic_path = output / 'diagnostic-config.json'
    if resume:
        if read(diagnostic_path) != diagnostic:
            raise ValueError('Frozen manufactured diagnostic changed')
    else:
        write(diagnostic_path, diagnostic)
    runtime = read(ROOT / config['runtime'])
    env = environment(runtime)
    cpu = sorted(os.sched_getaffinity(0))[-1]
    all_models = ['C53', 'C44', 'C37'] + [row['key'] for row in campaign['rows']]
    state = dict(status='running', completed=[], active=None, cuda_initialized=False,
        worker_count=1, available_host_bytes_before=available, source_sha256=sha(__file__), workers=[], reused_outputs=[])
    status_path = REPORTS / 'evaluation.json'
    if resume:
        previous = read(status_path)
        if previous['status'] != 'stopped' or not all(row['reaped'] for row in previous['workers']):
            raise ValueError('Resume requires a stopped evaluation with all prior workers reaped')
        attempt = 1
        while (REPORTS / f'evaluation-attempt-{attempt:02d}.json').exists():
            attempt += 1
        write(REPORTS / f'evaluation-attempt-{attempt:02d}.json', previous)
    def log_path(name):
        path = output / name
        attempt = 1
        while path.exists():
            attempt += 1
            path = output / (Path(name).stem + f'-attempt-{attempt:02d}.log')
        return path
    entries = {row['id']:row for row in read(frozen)['models']}
    def retained(path, identifier, *, fitted=False):
        if not path.exists():
            return False
        value = read(path)
        if value['status'] != 'passed':
            raise ValueError('Retained numerical result differs: ' + str(path))
        if 'material_sha256' in value:
            if value['material_sha256'] != sha(frozen):
                raise ValueError('Retained material identity differs')
        elif value['config_sha256'] != sha(diagnostic_path):
            raise ValueError('Retained diagnostic configuration differs')
        if value['weights_sha256'] != entries[identifier]['weights_sha256'] or value['cuda_initialized']:
            raise ValueError('Retained checkpoint identity or CPU contract differs')
        if fitted and (value['cases'] != read(bank_path)['cases']
                or value['amplitude_rms_pa'] != read(bank_path)['amplitude_rms_pa']):
            raise ValueError('Retained fitting cases differ from the common bank')
        state['reused_outputs'].append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        return True
    def stop(*_):
        raise KeyboardInterrupt('Evaluation interrupted; reaping the owned CPU worker')
    signal.signal(signal.SIGTERM, stop)
    try:
        bank_path = output / 'fitting-bank.json'
        if not bank_path.exists():
            run_worker([runtime['python'], '-B', HERE / 'fitted.py', 'freeze',
                '--config', config_path, '--material', frozen, '--cpu', cpu, '--output', bank_path],
                log_path('fitting-bank.log'), env, state['workers'])
        bank = read(bank_path)
        if bank['config_sha256'] != sha(config_path) or bank['material_sha256'] != sha(frozen):
            raise ValueError('Shared fitting bank changed')
        if resume and (output / 'C53-fitted.json').exists():
            parity = output / 'C53-fitted-common-parity.json'
            if not parity.exists():
                run_worker([runtime['python'], '-B', HERE / 'fitted.py', 'evaluate', '--bank', bank_path,
                    '--config', config_path, '--material', frozen, '--model', 'C53', '--cpu', cpu, '--output', parity],
                    log_path('C53-fitted-common-parity.log'), env, state['workers'])
            if read(parity)['measurements'] != read(output / 'C53-fitted.json')['measurements']:
                raise ValueError('Common bank does not exactly reproduce the original C53 fitted baseline')
            state['common_bank_control_exact'] = True
        for identifier in all_models:
            state['active'] = identifier
            write(status_path, state)
            if not retained(output / identifier / 'snapshots.json', identifier):
                run_worker([runtime['python'], '-B', ROOT / 'experiments/pressure-four-pass-distillation-v1/cpu.py',
                    'snapshots', '--material', frozen, '--id', identifier, '--cpu', cpu, '--output', output / identifier],
                    log_path(identifier + '-snapshots.log'), env, state['workers'])
            if not retained(output / (identifier + '-fitted.json'), identifier, fitted=True):
                run_worker([runtime['python'], '-B', HERE / 'fitted.py', 'evaluate', '--bank', bank_path,
                    '--config', config_path, '--material', frozen, '--model', identifier, '--cpu', cpu,
                    '--output', output / (identifier + '-fitted.json')], log_path(identifier + '-fitted.log'), env, state['workers'])
            # The retained C53 defines the existing manufactured deployment guard.
            # Older checkpoints do not contain the later band-feedback helper.
            if identifier not in ('C44', 'C37') and not retained(output / (identifier + '-diagnosis.json'), identifier):
                run_worker([runtime['python'], '-B', ROOT / 'experiments/pressure-low-mode-tail-v1/diagnose.py',
                    '--config', diagnostic_path, '--model', identifier, '--cpu', cpu,
                    '--output', output / (identifier + '-diagnosis.json')],
                    log_path(identifier + '-diagnosis.log'), env, state['workers'])
            state['completed'].append(identifier)
            write(status_path, state)
            print('Evaluated', identifier, flush=True)
        report(campaign, output)
        state.update(status='completed', active=None)
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        raise
    finally:
        write(status_path, state)


def report(campaign, output):
    def snapshot(identifier):
        return {row['passes']: row for row in read(output / identifier / 'snapshots.json')['summary']}
    def fitting(identifier):
        return read(output / (identifier + '-fitted.json'))
    def manufactured(identifier):
        return [row for row in read(output / (identifier + '-diagnosis.json'))['rows'] if row['passes'] == 8]
    def per_case(data):
        return {(row['case'], row['passes']): row['relative_equation_residual'] for row in data['measurements']}
    reference = snapshot('C53')
    manufactured_reference = sum(row['relative_equation_residual'] <= 1e-6 for row in manufactured('C53'))
    rows = []
    evidence = {str(path.relative_to(ROOT)): sha(path) for path in
        [output / 'material.json', output / 'evaluation-config.json', output / 'diagnostic-config.json']}
    baselines = {}
    for identifier in ('C53', 'C44', 'C37'):
        baselines[identifier] = dict(real=snapshot(identifier),
            fitted={row['passes']: row for row in fitting(identifier)['summary']})
        for path in [output / identifier / 'snapshots.json', output / (identifier + '-fitted.json')]:
            evidence[str(path.relative_to(ROOT))] = sha(path)
    evidence[str((output / 'C53-diagnosis.json').relative_to(ROOT))] = sha(output / 'C53-diagnosis.json')
    for item in campaign['rows']:
        identifier = item['key']
        own = snapshot(item['initialization'])
        real = snapshot(identifier)
        fit = fitting(identifier)
        fit_summary = {row['passes']: row for row in fit['summary']}
        original_fit = per_case(fitting(item['initialization']))
        candidate_fit = per_case(fit)
        capacity_checks = dict(
            four_pass_all_fitted=fit_summary[4]['equation_passes'] == 8,
            eight_pass_all_fitted=fit_summary[8]['equation_passes'] == 8,
            one_pass_no_case_worse=all(candidate_fit[c, 1] <= original_fit[c, 1] for c in range(8)),
            two_pass_no_case_worse=all(candidate_fit[c, 2] <= original_fit[c, 2] for c in range(8)))
        cases = manufactured(identifier)
        receipt_path = DIRECTORY / (identifier + '-study-01/receipt.json')
        receipt = read(receipt_path)
        training = receipt['training_receipt']
        history = read(receipt_path.parent / 'restore/history.json')
        regression_checks = dict(
            four_pass_all_128=real[4]['equation_passes'] == 128,
            eight_pass_all_128=real[8]['equation_passes'] == 128,
            earlier_pass_worst_no_worse_than_C53=all(real[n]['worst_equation_residual'] <= reference[n]['worst_equation_residual'] for n in (1, 2)),
            earlier_pass_worst_no_worse_than_own_start=all(real[n]['worst_equation_residual'] <= own[n]['worst_equation_residual'] for n in (1, 2)),
            four_pass_worst_no_worse_than_C53=real[4]['worst_equation_residual'] <= reference[4]['worst_equation_residual'],
            manufactured_eight_pass_no_worse_than_C53=sum(r['relative_equation_residual'] <= 1e-6 for r in cases) >= manufactured_reference)
        rows.append(dict(key=identifier, initialization=item['initialization'], mode=item['mode'],
            capacity_checks=capacity_checks, capacity_gate_passed=all(capacity_checks.values()),
            regression_checks=regression_checks, eligible_for_independent_scenes=all(capacity_checks.values()) and all(regression_checks.values()),
            fitted=fit_summary, real=real,
            training=dict(optimizer_updates=receipt['optimizer_updates'],
                trainer_elapsed_s=training['elapsed_s'], managed_elapsed_s=receipt['elapsed_s'],
                training_loop_s=history['training_finished_unix']-history['training_started_unix'],
                optimizer_work=history['optimizer_work'], peak_host_rss_bytes=training['runtime']['peak_rss_bytes'],
                peak_cuda_allocated_bytes=training['execution']['peak_allocated_bytes']),
            manufactured_eight_pass_qualified=sum(r['relative_equation_residual'] <= 1e-6 for r in cases),
            manufactured_eight_pass_median_low_mode_error_fraction=statistics.median(r['low_mode_pressure_error_fraction'] for r in cases)))
        for path in [output / identifier / 'snapshots.json', output / (identifier + '-fitted.json'),
                     output / (identifier + '-diagnosis.json'), receipt_path]:
            evidence[str(path.relative_to(ROOT))] = sha(path)
    paired = []
    indexed = {row['key']: row for row in rows}
    for initialization in ('C53', 'C44', 'C37'):
        stem = initialization.lower()
        d = indexed[stem + '-d']
        for comparator in ('a', 'u', 'start'):
            other = baselines[initialization] if comparator == 'start' else indexed[stem + '-' + comparator]
            other_key = initialization if comparator == 'start' else other['key']
            fit_d = per_case(fitting(d['key']))
            fit_other = per_case(fitting(other_key))
            paired.append(dict(initialization=initialization, candidate=d['key'], comparator=other_key,
                four_pass_fitted_worst_ratio=d['fitted'][4]['worst_equation_residual'] / other['fitted'][4]['worst_equation_residual'],
                four_pass_real_worst_ratio=d['real'][4]['worst_equation_residual'] / other['real'][4]['worst_equation_residual'],
                four_pass_real_median_ratio=d['real'][4]['median_equation_residual'] / other['real'][4]['median_equation_residual'],
                four_pass_fitted_cases_improved=sum(fit_d[c,4] < fit_other[c,4] for c in range(8)),
                four_pass_fitted_cases_worsened=sum(fit_d[c,4] > fit_other[c,4] for c in range(8))))
    result = dict(status='completed', baseline_retained='C53', rows=rows, paired=paired, baselines=baselines,
        eligible_for_independent_scenes=[row['key'] for row in rows if row['eligible_for_independent_scenes']],
        actual_optimizer_updates=sum(row['training']['optimizer_updates'] for row in rows),
        actual_network_forwards=sum(row['training']['optimizer_work']['network_forwards'] for row in rows),
        manufactured_C53_eight_pass_qualified=manufactured_reference, model_promoted=False,
        unused_holdouts_consumed=0, source_sha256=sha(__file__), evidence=evidence)
    write(REPORTS / 'results.json', result)
    lines = ['# Detached correction training results', '',
        'Nine fits from three starting checkpoints. C53 remains the retained reference. No model was promoted.', '',
        '| Start | Arm | Fitted four pass | Worst fitted residual | Real four pass | Real eight pass | Capacity gate |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['initialization']} | {row['mode']} | {row['fitted'][4]['equation_passes']}/8 | {row['fitted'][4]['worst_equation_residual']:.3g} | {row['real'][4]['equation_passes']}/128 | {row['real'][8]['equation_passes']}/128 | {'pass' if row['capacity_gate_passed'] else 'fail'} |")
    lines.extend(['', 'U is the existing fitting objective; A adds attached intermediate supervision; D detaches that auxiliary history.', '',
        'Capacity success is a specialist fitting result. The JSON report retains separate real and manufactured regression checks, paired comparisons, and source identities.', '',
        '[Full results and qualification gates](results.json)', ''])
    (REPORTS / 'results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    evaluate(resume=args.resume)
