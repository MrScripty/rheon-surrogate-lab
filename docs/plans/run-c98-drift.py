"""Record C98 on the existing C95 drift protocol without changing sealed studies."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
LAB = BASE.parent.parent
OUTPUT = BASE / 'chart-diagnostics/pressure-methods-v2/c98-drift-01'
EXPECTED_WEIGHT = '17bfb2d03e2e411a8d9a6eae57166315aca7dcb88cbc146c2c49639b82f53222'
sys.path.insert(0, str(LAB / 'experiments/pressure-four-pass-distillation-v1'))
from common import environment, read, sha, write


def main():
    if OUTPUT.exists():
        raise ValueError('Fresh diagnostic directory required; do not overwrite evidence')
    runtime = read(LAB / 'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    evaluation = LAB / 'runs/pressure-methods-v2/replicate-evaluation-01/v2-num6-s25/results.json'
    if read(evaluation)['entry']['weights_sha256'] != EXPECTED_WEIGHT:
        raise ValueError('C98 checkpoint identity differs')
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'controller-source.py').write_bytes(Path(__file__).read_bytes())
    runner = LAB / 'experiments/pressure-methods-v2/drift_episode.py'
    build = LAB / 'runs/pressure-methods-v2/harder-drift-build-02'
    jobs = [(0, 512, 20), (1, 512, 20), (4, 512, 20), (0, 7200, 21)]
    specification = dict(schema='chart-supplemental-drift-v1', model='C98',
        weights_sha256=EXPECTED_WEIGHT, evaluation=os.path.relpath(evaluation, BASE),
        evaluation_sha256=sha(evaluation), runner_sha256=sha(runner),
        build=os.path.relpath(build, BASE), build_receipt_sha256=sha(build / 'build-receipt.json'),
        source_sha256=sha(__file__), kind='neural', passes=7,
        scope='Matched C95 scenes and horizons; dual-track drift diagnostic',
        jobs=[dict(scene=s, steps=n, cpu=c) for s, n, c in jobs])
    write(OUTPUT / 'specification.json', specification)

    def episode(job):
        scene, steps, cpu = job
        destination = OUTPUT / f'scene{scene}-{steps}'
        command = [runtime['python'], '-B', str(runner), '--build', str(build),
            '--scene', str(scene), '--steps', str(steps), '--kind', 'neural', '--passes', '7',
            '--cpu', str(cpu), '--timeout', '1800', '--evaluation', str(evaluation),
            '--output', str(destination)]
        if scene == 4:
            command += ['--reservation', str(BASE / 'pressure-methods-v2/drift-reservation-03.json')]
        log = OUTPUT / f'scene{scene}-{steps}.log'
        tick = time.monotonic()
        child = None
        try:
            with log.open('xb') as stream:
                child = subprocess.Popen(command, env=environment(runtime), cwd=LAB,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait(timeout=1860)
            if code:
                raise RuntimeError(f'Episode failed with return code {code}; inspect {log}')
            result = read(destination / 'results.json')
            if not result['native_process']['reaped']:
                raise ValueError('Native child was not reaped')
            row = dict(path=os.path.relpath(destination / 'results.json', LAB),
                sha256=sha(destination / 'results.json'), scene_id=scene, steps=steps,
                reservation=os.path.relpath(BASE / ('pressure-methods-v2/drift-reservation-03.json'
                    if scene == 4 else 'pressure-methods-v2/drift-reservation.json'), BASE),
                status=result['status'], elapsed_s=time.monotonic()-tick,
                pid=child.pid, returncode=code, reaped=child.poll() is not None,
                log_sha256=sha(log), command=command)
            print(json.dumps({k:row[k] for k in ['scene_id','steps','status','elapsed_s']}), flush=True)
            return row
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()

    # Two independent CPU cores; the three short scenes run serially on core 20.
    def short_scenes():
        return [episode(job) for job in jobs[:3]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        shorts = pool.submit(short_scenes)
        long = pool.submit(episode, jobs[3])
        rows = shorts.result() + [long.result()]
    write(OUTPUT / 'manifest.json', dict(schema=specification['schema'], status='completed',
        specification_sha256=sha(OUTPUT / 'specification.json'), model='C98',
        weights_sha256=EXPECTED_WEIGHT, drift=rows, owned_children_reaped=all(r['reaped'] for r in rows)))


if __name__ == '__main__':
    main()
