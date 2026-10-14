#!/usr/bin/env python3
"""Frozen cached-GroupNorm development diagnostic; no model replay or state advance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import re
import signal
import stat
import statistics
import sys
import time
import types

STARTED = time.monotonic()
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

MIB = 1024 * 1024
REPORT_LIMITS = {"input-proof.json": MIB, "population.json": 64 * 1024,
                 "cases.jsonl": 8 * MIB, "evaluation.json": 16 * MIB,
                 "failure.json": MIB}
METHODS = ("learned_jacobi", "temporal_full_budget", "temporal_jacobi", "zero_jacobi")
RAW_METHODS = ("reference", "raw_learned", "raw_temporal", "raw_zero")
LAB = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check_deadline(deadline):
    require(time.monotonic() < deadline, "workflow_deadline")
    # Linux ru_maxrss is KiB; this is an observed high-water policy, not containment.
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 4 * 1024**3,
            "adapter_observed_rss_limit")


def precheck_output(path):
    path = Path(path).absolute()
    require(path.resolve() == path and path.parent == Path('/tmp') and not path.exists()
            and not path.is_symlink(), 'fresh_direct_tmp_output_required')
    return path


class Outputs:
    """Private single-writer directory; only evaluation.json is successful authority.

    Pending/incomplete files preserve interrupted evidence. No consumer may accept
    them or inspect this private staging directory before the command terminates.
    """
    def __init__(self, path, deadline):
        self.path = precheck_output(path)
        check_deadline(deadline)
        self.path.mkdir(mode=0o700)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(self.fd)
        self.identity = (info.st_dev, info.st_ino)
        self.deadline = deadline
        self.sizes = {}
        self.created = set()
        self.journal_fd = None

    def _directory(self):
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(self.fd)
            require((info.st_dev, info.st_ino) == self.identity, 'output_directory_identity')
        return self.fd

    def _reserve(self, name, addition):
        require(name in REPORT_LIMITS, 'undeclared_output')
        check_deadline(self.deadline)
        require(self.sizes.get(name, 0) + addition <= REPORT_LIMITS[name], 'output_file_limit:' + name)
        require(sum(self.sizes.values()) + addition <= 32 * MIB, 'aggregate_output_limit')
        self.sizes[name] = self.sizes.get(name, 0) + addition

    def write(self, name, value):
        blob = canonical(value) + b'\n'
        require(name not in self.sizes, 'output_already_published:' + name)
        self._reserve(name, len(blob))
        staged = 'evaluation.pending.json' if name == 'evaluation.json' else name
        directory = self._directory()
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        self.created.add(staged)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
        check_deadline(self.deadline)
        if name == 'evaluation.json':
            # Exclusive hard-link publication never overwrites a target. The
            # private directory has one writer and no concurrent path mutation.
            os.link(staged, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            self.created.add(name)
            os.unlink(staged, dir_fd=directory)
            self.created.remove(staged)
            os.fsync(directory)
            check_deadline(self.deadline)

    def append(self, record):
        blob = canonical(record) + b'\n'
        self._reserve('cases.jsonl', len(blob))
        directory = self._directory()
        if self.journal_fd is None:
            self.journal_fd = os.open('cases.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                      0o600, dir_fd=directory)
            self.created.add('cases.jsonl')
        view = memoryview(blob)
        while view:
            check_deadline(self.deadline)
            count = os.write(self.journal_fd, view)
            require(count > 0, 'journal_write_stalled')
            view = view[count:]
        os.fsync(self.journal_fd)
        os.fsync(directory)
        check_deadline(self.deadline)

    def revoke_evaluation(self):
        """Retain own failed evidence; remove successful authority on any failure."""
        directory = self._directory()
        for name in ('evaluation.json', 'evaluation.pending.json'):
            if name in self.created:
                if 'evaluation.incomplete.json' not in self.created:
                    # All names are private single-writer state, not arbitrary
                    # paths. No existing result or prior evidence is replaced.
                    require(not (self.path/'evaluation.incomplete.json').exists(), 'incomplete_output_exists')
                    os.rename(name, 'evaluation.incomplete.json', src_dir_fd=directory, dst_dir_fd=directory)
                    self.created.add('evaluation.incomplete.json')
                else:
                    os.unlink(name, dir_fd=directory)
                self.created.remove(name)
        os.fsync(directory)

    def close(self):
        if self.journal_fd is not None:
            descriptor, self.journal_fd = self.journal_fd, None
            os.close(descriptor)
        if self.fd is not None:
            descriptor, self.fd = self.fd, None
            os.close(descriptor)


def stats(values):
    return {"count": len(values), "sum": sum(values), "minimum": min(values),
            "median": statistics.median(values), "maximum": max(values)} if values else {
                "count": 0, "sum": 0, "minimum": None, "median": None, "maximum": None}


def attempts(record, method):
    policy = record[method]
    if method in ('learned_jacobi', 'temporal_jacobi'):
        return [(phase, policy[phase]) for phase in ('warm', 'reset') if policy[phase] is not None]
    return [('full', policy)]


def completed_iterations(attempt):
    error = attempt['solver_error']
    # Native breakdown's iteration is an index; it is not completed work.
    if error is not None and not error.startswith(('pressure iteration limit ', 'pressure solve cancelled after ')):
        return None
    return attempt['iterations']


def final_quality(record, method):
    policy = record[method]
    if method in ('temporal_full_budget', 'zero_jacobi'):
        return policy['quality']
    if policy['reset'] is not None:
        return policy['reset']['quality']
    if policy['warm'] is not None:
        return policy['warm']['quality']
    return record['raw_learned' if method == 'learned_jacobi' else 'raw_temporal']


def aggregate(records):
    summary = {'cases': len(records), 'raw': {}, 'methods': {}}
    for method in RAW_METHODS:
        failures = {}
        for record in records:
            for failure in record[method]['failures']:
                failures[failure] = failures.get(failure, 0) + 1
        summary['raw'][method] = {'qualified': sum(record[method]['qualified'] for record in records),
                                  'failure_gates': failures}
    for method in METHODS:
        all_attempts = [(record, phase, attempt) for record in records
                        for phase, attempt in attempts(record, method)]
        known = [completed_iterations(a) for _, _, a in all_attempts if completed_iterations(a) is not None]
        warm = [completed_iterations(a) for _, phase, a in all_attempts if phase == 'warm' and completed_iterations(a) is not None]
        reset = [completed_iterations(a) for _, phase, a in all_attempts if phase == 'reset' and completed_iterations(a) is not None]
        wasted = [completed_iterations(a) for record, phase, a in all_attempts
                  if phase == 'warm' and record[method]['reset'] is not None and completed_iterations(a) is not None]
        policies = [r[method] for r in records]
        entry = {'qualified': sum(p['qualified'] for p in policies),
                 'attempts': len(all_attempts), 'completed_iterations_known': stats(known),
                 'completed_iterations_unknown_attempts': len(all_attempts) - len(known),
                 'warm_completed_iterations_known': stats(warm), 'reset_completed_iterations_known': stats(reset),
                 'wasted_warm_completed_iterations_known': stats(wasted),
                 'zero_iteration_attempts': sum(completed_iterations(a) == 0 for _, _, a in all_attempts),
                 'solver_errors': sum(a['solver_error'] is not None for _, _, a in all_attempts),
                 'solver_workspace_guess_hash_s': stats([a['solver_s'] for _, _, a in all_attempts]),
                 'post_solve_verification_s': stats([a['verification_s'] for _, _, a in all_attempts])}
        if method in ('learned_jacobi', 'temporal_jacobi'):
            outcomes = {}
            rejections = {}
            for p in policies:
                outcomes[p['outcome']] = outcomes.get(p['outcome'], 0) + 1
                reason = p['proposal_rejection']
                if reason is not None:
                    rejections[reason] = rejections.get(reason, 0) + 1
            entry.update(outcomes=outcomes, proposal_rejections=rejections,
                         any_pcg=sum(p['any_pcg'] for p in policies),
                         resets=sum(p['reset_fallback'] for p in policies),
                         corrected_path_s_excludes_initial_raw_checks=stats([p['elapsed_s'] for p in policies]))
        quality = [final_quality(r, method) for r in records]
        for field in ('residual_l2', 'residual_max', 'relative_residual', 'predicted_divergence_max_s_inv',
                      'actual_divergence_max_s_inv', 'actual_divergence_rms_s_inv', 'net_flux_m3_s', 'courant'):
            entry[field] = stats([q[field] for q in quality if field in q and q[field] is not None])
        entry['actual_divergence_l2_derived_rms_times_sqrt4096'] = stats([
            q['actual_divergence_rms_s_inv'] * 64 for q in quality if 'actual_divergence_rms_s_inv' in q])
        entry['absolute_net_flux_m3_s'] = stats([abs(q['net_flux_m3_s']) for q in quality if q.get('net_flux_m3_s') is not None])
        entry['wall_flux_exact_zero'] = sum(q.get('wall_normal_flux_exact_zero') is True for q in quality)
        summary['methods'][method] = entry
    summary['recorded_restore_inference_s_separate_historical_execution'] = stats([
        r['recorded_model_inference_s'] for r in records])
    summary['all_methods_native_elapsed_s'] = stats([r['native_elapsed_s'] for r in records])
    summary['all_methods_process_wall_s'] = stats([r['process']['wall_s'] for r in records])
    summary['cached_raw_pressure_error_l2_pa'] = stats([r['raw_pressure_error_l2_pa'] for r in records if 'raw_pressure_error_l2_pa' in r])
    summary['cached_raw_pressure_relative_l2'] = stats([r['raw_pressure_relative_l2'] for r in records if 'raw_pressure_relative_l2' in r])
    summary['zero_residual_base_retained_cases'] = sum(r.get('zero_residual_base_retained') is True for r in records)
    return summary


def summaries(records):
    return {'all': aggregate(records),
            'split': {split: aggregate([r for r in records if r['split'] == split]) for split in ('train', 'validation')},
            'frame': {str(frame): aggregate([r for r in records if r['frame'] == frame]) for frame in (0, 23)},
            'trajectory_pairs': {tid: aggregate([r for r in records if r['trajectory_id'] == tid])
                                 for tid in dict.fromkeys(r['trajectory_id'] for r in records)}}


def source_freeze(path, expected_sha256):
    path = Path(path).absolute()
    require(path.resolve() == path and not path.is_symlink() and path.is_file(), 'freeze_regular_file')
    blob = bounded_source_bytes(path, MIB)
    require(hashlib.sha256(blob).hexdigest() == expected_sha256, 'reviewed_freeze_identity')
    # Exact byte identity binds this authored freeze before any new module import.
    freeze = json.loads(blob)
    require(type(freeze) is dict and set(freeze) == {
        'schema', 'implementation_commit', 'scientific_source_commit', 'source_sha256',
        'input_freeze_sha256', 'input_paths', 'limits'}, 'freeze_fields')
    require(freeze['schema'] == 'rheon-groupnorm-native-development-source-freeze-v1', 'freeze_schema')
    require(freeze['scientific_source_commit'] == 'fd3a6009e5805978a29ed1df7389ad226a40a914', 'freeze_candidate')
    require(type(freeze['implementation_commit']) is str and re.fullmatch('[0-9a-f]{40}', freeze['implementation_commit']), 'freeze_commit')
    require(type(freeze['input_freeze_sha256']) is str and re.fullmatch('[0-9a-f]{64}', freeze['input_freeze_sha256']), 'freeze_inputs_identity')
    require(type(freeze['source_sha256']) is dict and type(freeze['input_paths']) is dict
            and set(freeze['input_paths']) == {'training_artifacts', 'restored_artifacts', 'capture', 'packet', 'binary'}
            and all(type(v) is str for v in freeze['input_paths'].values()), 'freeze_inventory')
    require(type(freeze['limits']) is dict and all(type(v) is int for v in freeze['limits'].values()), 'freeze_limit_types')
    require(freeze['limits'] == {'verify_s': 60, 'evaluate_s': 600, 'input_bytes': 256*MIB,
                                'output_bytes': 32*MIB, 'observed_rss_bytes': 4*1024**3}, 'freeze_limits')
    verify_sources(freeze)
    for name in ('evaluate.py', 'admission.py', 'process.py'):
        require('experiments/global-pressure-v1/groupnorm-native-dev-v1/' + name in freeze['source_sha256'], 'incomplete_source_freeze')
    return freeze


def verify_sources(freeze):
    for name, expected in freeze['source_sha256'].items():
        require(type(name) is str and type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'frozen_source_identity')
        target = LAB / name
        require(target.resolve() == target and LAB in target.parents and not target.is_symlink() and target.is_file(), 'frozen_source_path')
        require(hashlib.sha256(bounded_source_bytes(target, 2*MIB)).hexdigest() == expected, 'frozen_source_drift:' + name)


def import_frozen(freeze, name):
    source_name = 'experiments/global-pressure-v1/groupnorm-native-dev-v1/' + name + '.py'
    blob = bounded_source_bytes(LAB/source_name, 2*MIB)
    require(hashlib.sha256(blob).hexdigest() == freeze['source_sha256'][source_name], 'module_source_drift:' + name)
    module = types.ModuleType('gn_native_frozen_' + name)
    module.__file__ = str(LAB/source_name)
    sys.modules[module.__name__] = module
    try:
        exec(compile(blob, module.__file__, 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def _execute(command, paths, output, freeze, state, *, admission_api=None, process_api=None):
    if admission_api is None:
        admission_api = import_frozen(freeze, 'admission')
    if process_api is None:
        process_api = import_frozen(freeze, 'process')
    limit = 60 if command == 'verify' else 600
    deadline = STARTED + limit
    work_deadline = deadline - 1
    admitted = None
    records = []
    children = []
    outputs = None
    try:
        precheck_output(output)
        require({k: str(Path(v).absolute()) for k, v in paths.items()} == freeze['input_paths'], 'frozen_input_locator')
        admitted = admission_api.admit(paths, work_deadline)
        check_deadline(deadline)
        outputs = Outputs(output, deadline)
        state["outputs"] = outputs
        require(admitted['proof'].get('input_bytes_read', 0) + SOURCE_BYTES_READ <= 256*MIB, 'total_input_read_limit')
        outputs.write('input-proof.json', admitted['proof'])
        if command == 'verify':
            require(outputs.sizes['input-proof.json'] <= MIB, 'verify_report_limit')
            outputs.close()
            outputs = None
            return {'status': 'verified_local_inputs', 'cases': len(admitted['population']),
                    'workflow_elapsed_s_through_publication': time.monotonic() - STARTED,
                    'native_calls': 0, 'output': str(output)}
        require(len(admitted['population']) == len(admitted['requests']) == len(admitted['inference']) == 32,
                'complete_fixed32_population')
        outputs.write('population.json', admitted['population'])
        identity, measurement = process_api.native_call(paths['binary'], '--identity', None, work_deadline, 10)
        children.append({'purpose': 'initial_identity', **measurement})
        require(identity == admitted['identity'], 'native_initial_identity')
        for index, (case, request) in enumerate(zip(admitted['population'], admitted['requests'])):
            check_deadline(deadline)
            result, measurement = process_api.native_call(paths['binary'], '--evaluate', request, work_deadline, 60)
            record = {**case, **result, 'recorded_model_inference_s': admitted['inference'][index],
                      'recorded_inference_scope': 'earlier cold restore; no fresh inference', 'process': measurement}
            # Retain the actual response before validation/fail-stop; invalid response
            # never becomes an accepted record or a silent numerical rescue.
            outputs.append(record)
            records.append(record)
            admission_api.validate_result(request, result, case)
            require(all(result[m]['qualified'] for m in METHODS), 'terminal_numerical_failure:' + case['case_id'])
        identity, measurement = process_api.native_call(paths['binary'], '--identity', None, work_deadline, 10)
        children.append({'purpose': 'final_identity', **measurement})
        require(identity == admitted['identity'], 'native_final_identity')
        admission_api.verify_unchanged(admitted, work_deadline)
        verify_sources(freeze)
        require(admitted['proof'].get('input_bytes_read', 0) + SOURCE_BYTES_READ <= 256*MIB, 'total_input_read_limit')
        check_deadline(deadline)
        report = {'schema': 'rheon-groupnorm-native-development-evaluation-v1', 'status': 'complete_diagnostic',
                  'scope': 'fixed32 development teacher snapshots; no autonomous rollout/final qualification/acceleration claim',
                  'case_population': admitted['population'], 'cases': records, 'summaries': summaries(records),
                  'input_proof': admitted['proof'], 'source_freeze': freeze, 'identity_processes': children,
                  'source_admission_bytes_read': SOURCE_BYTES_READ,
                  'workflow_clock_scope': 'from module STARTED after stdlib imports; includes source/admission and final publication, excludes initial interpreter/stdlib startup',
                  'workflow_elapsed_s_before_final_publication': time.monotonic() - STARTED,
                  'total_declared_input_bytes_read': admitted['proof'].get('input_bytes_read', 0) + SOURCE_BYTES_READ,
                  'adapter_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  'cost_scope': 'new local workflow includes admission, all native methods/checks/process lifecycle and publication; historical restore inference is separately recorded',
                  'unavailable': ['initial raw-check/setup individual timers', 'standalone candidate enclosing wall/speedup',
                                  'post-PCG pressure error', 'corrected velocity error', 'rollout/final/GPU metrics'],
                  'state_advanced': False, 'fresh_model_inference': False, 'new_optimizer_updates': 0}
        publication_started = time.monotonic()
        outputs.write('evaluation.json', report)
        outputs.close()
        outputs = None
        check_deadline(deadline)
        return {'status': 'complete_diagnostic', 'cases': len(records),
                'raw_qualified': report['summaries']['all']['raw']['raw_learned']['qualified'],
                'learned_qualified': report['summaries']['all']['methods']['learned_jacobi']['qualified'],
                'workflow_elapsed_s_through_publication': time.monotonic() - STARTED,
                'final_report_publication_s': time.monotonic() - publication_started,
                'output': str(Path(output) / 'evaluation.json')}
    except BaseException as error:
        failure = {'schema': 'rheon-groupnorm-native-development-failure-v1', 'status': 'failed',
                   'observed_responses': len(records), 'operation': command, 'reason': str(error)[:4096],
                   'error_type': type(error).__name__, 'measurement': getattr(error, 'measurement', None),
                   'identity_processes': children, 'workflow_elapsed_s': time.monotonic() - STARTED,
                   'automatic_retry': False, 'state_advanced': False}
        if outputs is not None:
            outputs.revoke_evaluation()
            # After a spent deadline, writing a bounded failure is best effort;
            # inability to publish it remains failure, never success or retry.
            try:
                outputs.write('failure.json', failure)
            except BaseException as publication_error:
                failure['failure_publication_error'] = str(publication_error)[:1024]
        print(json.dumps(failure, sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)
        raise
    finally:
        if outputs is not None:
            outputs.close()


SOURCE_BYTES_READ = 0


def bounded_source_bytes(path, limit):
    global SOURCE_BYTES_READ
    path = Path(path).absolute()
    require(path.resolve() == path and not path.is_symlink(), 'source_regular_path')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit, 'source_file_bound')
        blob = stream.read(limit + 1)
    require(len(blob) == info.st_size and len(blob) <= limit, 'source_read_bound')
    SOURCE_BYTES_READ += len(blob)
    require(SOURCE_BYTES_READ <= 16*MIB, 'source_admission_read_bound')
    return blob


def execute(command, paths, output, freeze, *, admission_api=None, process_api=None,
            freeze_path=None, freeze_sha256=None):
    require(command in ('verify', 'evaluate'), 'unsupported_operation')
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'existing_workflow_watchdog')
    previous = signal.getsignal(signal.SIGALRM)
    state = {}
    def expired(_signal, _frame):
        raise ValueError('workflow_watchdog_deadline')
    signal.signal(signal.SIGALRM, expired)
    try:
        remaining = STARTED + (60 if command == 'verify' else 600) - 1 - time.monotonic()
        require(remaining > 0, 'workflow_deadline')
        signal.setitimer(signal.ITIMER_REAL, remaining)
        precheck_output(output)
        if freeze is None:
            freeze = source_freeze(freeze_path, freeze_sha256)
        result = _execute(command, paths, output, freeze, state,
                          admission_api=admission_api, process_api=process_api)
        check_deadline(STARTED + (60 if command == 'verify' else 600))
        result['workflow_elapsed_s_through_publication'] = time.monotonic() - STARTED
        return result
    except BaseException:
        # This owner still holds staging identity after inner close/return. An
        # outer deadline, close or alarm failure must revoke its complete name.
        if state.get('outputs') is not None:
            state['outputs'].revoke_evaluation()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if state.get('outputs') is not None:
            state['outputs'].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'evaluate'))
    for name in ('training-artifacts', 'restored-artifacts', 'capture', 'packet', 'binary', 'freeze', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ('training_artifacts', 'restored_artifacts', 'capture', 'packet', 'binary')}
    result = execute(args.command, paths, args.output, None,
                     freeze_path=args.freeze, freeze_sha256=args.freeze_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed", "reason": str(error)[:4096],
                          "error_type": type(error).__name__, "automatic_retry": False}),
              file=sys.stderr, flush=True)
        raise SystemExit(1)
