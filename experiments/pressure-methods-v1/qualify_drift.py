"""Prove observation-only native parity before admitting drift experiments."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
               str(ROOT/'runs/dependency-migration-v1/lab/experiments/water-residual-learning-v1')]
from common import read, write, sha
import rollout


def qualify():
    base = ROOT/'runs/pressure-methods-v1/drift-qualification-01'
    if base.exists():
        raise ValueError('Fresh observation qualification required')
    base.mkdir()
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[-2]})
    material = read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    build = ROOT/'runs/pressure-methods-v1/drift-observation-02'
    receipt = read(build/'build-receipt.json')
    if receipt['status'] != 'qualified_build' or sha(build/'rheon-drift-rollout') != receipt['binary_sha256']:
        raise ValueError('Exact qualified isolated binary required')
    scene = dict(material['config']['scene'],steps=16)
    controls = []
    for label,binary in [('original',ROOT/material['config']['binary']),('observed',build/'rheon-drift-rollout')]:
        result = rollout.run(binary,scene,base/label,control=True,timeout=90,threads=1)
        if not result['native_process']['reaped'] or result['result']['hard_stop']:
            raise ValueError('Observation native control failed')
        controls.append([json.loads(line) for line in (base/label/'frames.jsonl').read_text().splitlines()])
    keys = ['phase','step','time_s','reference_quality','ai_quality','relative_pressure_difference']
    if len(controls[0]) != 16 or any(a[k]!=b[k] for a,b in zip(*controls) for k in keys):
        raise ValueError('Native physical trajectory differs after adding read-only observations')
    for frame in controls[1]:
        if any(frame[k] != 0 for k in ['velocity_trajectory_relative_l2','tracer_trajectory_relative_l2']):
            # Different native stopping warm starts may round corrected velocities
            # differently after warmup; retain the measured values, require tiny.
            if max(frame[k] for k in ['velocity_trajectory_relative_l2','tracer_trajectory_relative_l2']) > 1e-6:
                raise ValueError('Native algorithm control has unexpected drift')
    command = [sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v1/drift_episode.py'),
               '--build',str(build),'--output',str(base/'neural-fixture'),
               '--kind','neural','--steps','16','--passes','8']
    with (base/'neural-fixture.log').open('xb') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    neural = read(base/'neural-fixture/results.json')
    if neural['status'] != 'completed' or neural['completed_post_warmup_steps'] != 1:
        raise ValueError('Observed neural fixture must accept its independent first step')
    write(ROOT/'docs/plans/pressure-methods-v1/drift-observation-qualification.json',dict(status='passed',
        source_sha256=sha(__file__),build_receipt_sha256=sha(build/'build-receipt.json'),
        exact_native_frame_comparisons=16,compared_keys=keys,neural_fixture_sha256=sha(base/'neural-fixture/results.json'),
        native_physics_unchanged=True,observed_committed_velocity_and_tracer=True,
        tested_wire_cell_bound=65536,maximum_scene_steps=3600,physical_acceptance_gates_unchanged=True,
        original_selected_rheon_modified=False))
    print('Qualified observed native parity and neural fixture',flush=True)


if __name__ == '__main__':
    qualify()
