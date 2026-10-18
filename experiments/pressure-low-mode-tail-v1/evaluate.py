"""Isolate restored CPU timings, then diagnose manufactured contraction in parallel."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0, str(PREVIOUS))
from common import environment, read, write
from cpu import material


def evaluate(config_path, projects, output):
    config = read(config_path)
    runtime = read(ROOT/config['runtime'])
    controller = read(projects/'controller-execution.json')
    if controller['status'] != 'completed' or not controller['all_owned_controllers_reaped']:
        raise ValueError('Managed controllers/readback must finish before isolated timings')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    if available < 16*1024**3:
        raise MemoryError('12 GiB host reserve plus two 2 GiB CPU workers required')
    output.mkdir(exist_ok=False)
    frozen = output/'material.json'
    material(config_path, projects, frozen, True)
    diagnostic_config = {**config, 'baseline_material':str(frozen.relative_to(ROOT))}
    write(output/'diagnostic-config.json', diagnostic_config)
    children, records = [], []
    started = time.perf_counter()
    status = 'failed'
    def stop(*_):
        raise KeyboardInterrupt('CPU evaluation cancelled')
    signal.signal(signal.SIGTERM, stop)
    def start(command, label):
        log = (output/(label+'.log')).open('xb')
        process = subprocess.Popen(command, env=environment(runtime), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        job = dict(process=process, log=log, command=command, label=label, started=time.perf_counter())
        children.append(job)
        return job
    def finish(job):
        code = job['process'].wait(timeout=180)
        job['log'].close()
        records.append(dict(label=job['label'], command=job['command'], returncode=code,
                            elapsed_s=time.perf_counter()-job['started'], reaped=True))
        if code:
            raise RuntimeError('CPU evaluation failed: '+job['label'])
    try:
        cpu = max(os.sched_getaffinity(0))
        identifiers = config['baselines']+[arm['model_id'] for arm in config['arms']]
        for identifier in identifiers:
            command = [runtime['python'], '-B', str(PREVIOUS/'cpu.py'), 'snapshots', '--material', str(frozen),
                       '--id', identifier, '--cpu', str(cpu), '--output', str(output/identifier)]
            finish(start(command, identifier+'-snapshots'))
            print('Isolated snapshots complete', identifier, flush=True)
        diagnostic_jobs = []
        cpus = sorted(os.sched_getaffinity(0))[-2:]
        for arm,cpu in zip(config['arms'], cpus):
            command = [runtime['python'], '-B', str(HERE/'diagnose.py'), '--config', str(output/'diagnostic-config.json'),
                       '--model', arm['model_id'], '--cpu', str(cpu), '--output', str(output/(arm['model_id']+'-diagnosis.json'))]
            diagnostic_jobs.append(start(command, arm['model_id']+'-diagnosis'))
        for job in diagnostic_jobs:
            finish(job)
        status = 'completed'
    finally:
        for job in children:
            process = job['process']
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            job['log'].close()
        write(output/'execution.json', dict(status=status, elapsed_s=time.perf_counter()-started, rows=records,
            available_host_bytes_before=available, all_owned_children_reaped=all(j['process'].poll() is not None for j in children),
            timing_policy='Four checkpoints measured serially, one CPU thread/core, after managed readback drained. Postfit manufactured diagnosis runs afterward with two independent one-core workers.',
            environment={k:v for k,v in environment(runtime).items() if k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES',
                'TULDOK_TORCH_DEVICE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'projects', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.config.resolve(), args.projects.resolve(), args.output.resolve())
