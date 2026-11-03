"""Five balanced serial CPU repeats on fixed requests from every fresh scene."""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time
from study_support import ROOT, read, write, sha, runtime, load_checkpoint, equation_error
from fresh_pressure_data import cases


def worker(args):
    os.sched_setaffinity(0, {23})
    tick = time.perf_counter()
    state = load_checkpoint(args.worker)
    torch = state.torch
    torch.set_num_threads(1)
    samples = cases(args.bank, torch, timing_only=True)
    inventory = state.data.helpers().state_inventory(state.network)
    predict = lambda sample: state.model.predict_pressure(state.network, sample, state.normalization)
    with torch.inference_mode():
        for _, sample in samples:
            predict(sample)
    print(json.dumps(dict(kind='ready', identifier=args.worker, setup_seconds=time.perf_counter()-tick,
                          requests=len(samples), entry=state.entry, source_sha256=sha(__file__),
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        if command['kind'] == 'stop':
            break
        order = list(range(len(samples)))
        random.Random(2026101957 + command['repeat']).shuffle(order)
        rows = []
        with torch.inference_mode():
            for _, sample in samples:
                predict(sample)
            for index in order:
                identity, sample = samples[index]
                tick = time.perf_counter()
                pressure = predict(sample)
                elapsed = time.perf_counter() - tick
                if not bool(torch.isfinite(pressure).all()) or float(pressure[0, 0, 0, 0, 0]) != 0:
                    raise ValueError('Pressure output contract failed')
                rows.append(dict(identity, seconds=elapsed, equation_error=equation_error(state, pressure, sample)))
        if inventory != state.data.helpers().state_inventory(state.network):
            raise ValueError('Timing changed weights')
        print(json.dumps(dict(kind='repeat', identifier=args.worker, repeat=command['repeat'], rows=rows,
                              threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                              cuda_initialized=torch.cuda.is_initialized())), flush=True)


def controller(args):
    args.output = args.output.resolve()
    args.bank = args.bank.resolve()
    selection = read(args.selection)
    if selection['status'] != 'frozen_before_reserved_capture' or read(args.bank)['selection_sha256'] != sha(args.selection):
        raise ValueError('Frozen selection and corresponding fresh capture required')
    if args.output.exists():
        raise ValueError('Fresh timing directory required')
    args.output.mkdir(parents=True)
    from common import environment
    selected = runtime()
    children, streams, results = {}, {}, []
    record = dict(status='running', active=None, completed=[], balanced_serial=True, concurrent_numerical_workers=False)
    def receive(child):
        line = child.stdout.readline()
        if not line:
            raise RuntimeError('CPU worker stopped; retained stderr contains diagnosis')
        return json.loads(line)
    try:
        for model in selection['models']:
            identifier = model['id']
            if sha(ROOT / model['entry_path']) != model['entry_sha256']:
                raise ValueError('Frozen model entry changed')
            stream = (args.output / (identifier + '-stderr.log')).open('xb')
            streams[identifier] = stream
            child = subprocess.Popen([selected['python'], '-B', str(Path(__file__).resolve()),
                '--worker', identifier, '--bank', str(args.bank)], cwd=ROOT, env=environment(selected),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stream, text=True)
            children[identifier] = child
            write(args.output / (identifier + '-setup.json'), receive(child))
        identifiers = list(children)
        for repeat in range(5):
            order = identifiers[repeat % len(identifiers):] + identifiers[:repeat % len(identifiers)]
            if repeat % 2:
                order.reverse()
            for identifier in order:
                record['active'] = dict(identifier=identifier, repeat=repeat, pid=children[identifier].pid)
                write(args.output / 'execution.json', record)
                child = children[identifier]
                child.stdin.write(json.dumps(dict(kind='time', repeat=repeat)) + '\n')
                child.stdin.flush()
                result = receive(child)
                results.append(result)
                path = args.output / f'{identifier}-r{repeat}.json'
                write(path, result)
                record['completed'].append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                record['active'] = None
                write(args.output / 'execution.json', record)
                print('Fresh pressure timing', identifier, 'repeat', repeat, flush=True)
        summaries = []
        for identifier in identifiers:
            for population in ('small', '32cubed_extension'):
                groups = [[row for row in result['rows'] if row['population'] == population]
                          for result in results if result['identifier'] == identifier]
                medians = [1000 * statistics.median(row['seconds'] for row in group) for group in groups]
                pooled = sorted(1000 * row['seconds'] for group in groups for row in group)
                summaries.append(dict(identifier=identifier, population=population, repeats=5,
                    cases_per_repeat=len(groups[0]), median_ms=statistics.median(medians),
                    min_repeat_median_ms=min(medians), max_repeat_median_ms=max(medians),
                    p95_ms=pooled[min(len(pooled)-1, int(.95*len(pooled)))], maximum_ms=max(pooled),
                    worst_equation_error=max(row['equation_error'] for group in groups for row in group)))
        write(args.output / 'summary.json', dict(status='completed', rows=summaries,
            bank_sha256=sha(args.bank), selection_sha256=sha(args.selection), source_sha256=sha(__file__),
            scope='Complete predict_pressure; sample construction, loading and independent equation check excluded',
            warmup='Entire64-request population before each repeat', cpu=23, threads=1, batch_size=1,
            reserved_scene_steps=[1,16,32,64], larger_grids_reported_separately=True))
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        for identifier, child in children.items():
            if child.poll() is None:
                try:
                    child.stdin.write(json.dumps(dict(kind='stop')) + '\n')
                    child.stdin.flush()
                    child.wait(timeout=10)
                except BaseException:
                    child.terminate()
                    child.wait(timeout=10)
            streams[identifier].close()
        record.update(active=None, owned_children=[dict(identifier=key, pid=child.pid, returncode=child.returncode,
                      reaped=child.poll() is not None) for key, child in children.items()])
        write(args.output / 'execution.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    for name in ('bank', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=name == 'bank')
    args = parser.parse_args()
    worker(args) if args.worker else controller(args)
