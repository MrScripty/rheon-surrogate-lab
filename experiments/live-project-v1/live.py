#!/usr/bin/env python3
"""Observe real frozen CPU capture/evaluation in Tuldok and Rheon's egui app."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import signal
import time

import numpy as np

from observer import Observer, Watch, atomic

LAB = Path(__file__).resolve().parents[2]
PROFILE = LAB/'experiments/local-pressure-v1'
sys.path.insert(0, str(PROFILE))
import capture
import contract
import evaluate


def trajectories(config):
    return [{**entry, 'extents_m': config['geometries_extents_m'][entry['geometry']]} for entry in config['trajectories']]


def generate(args, observer):
    config = contract.experiment(contract.decode(contract.read(args.config, contract.MIB)))
    entries = trajectories(config)
    seen = set()
    observer.publish('Generation', 'Generating native pressure/tracer frames; qualification pending.', total=len(entries)*24)
    def poll():
        newest = None
        for entry in entries:
            for index in range(24):
                key = (entry['trajectory_id'], index)
                if key in seen:
                    continue
                directory = args.output/entry['trajectory_id']
                fields = [(directory/f'frame-{index:03d}-{name}.bin', size) for name,size in (('tracer',16384),('pressure',32768))]
                if all(path.is_file() and path.stat().st_size == size for path,size in fields):
                    preview = observer.frame(args.output, entry, index)
                    seen.add(key); newest = preview
        if newest is not None:
            observer.publish('Generation', f'Observed {len(seen)} native frames; independent qualification pending.', len(seen), len(entries)*24, preview=newest)
    with Watch(poll):
        # Qualify a real one-frame fixture before complete development generation.
        fixture = args.output.with_name(args.output.name+'-fixture')
        capture.generate(config, args.binary, fixture, fixture=True)
        observer.publish('Generation', 'One-frame fixture independently qualified. Generating full development capture.', total=len(entries)*24)
        result = capture.generate(config, args.binary, args.output)
    poll()
    atomic(observer.root/'capture-manifest.json', contract.decode(contract.read(args.output/'manifest.json',4*contract.MIB)))
    observer.preview['pressure_label'] = 'Native reference pressure (Pa); capture independently verified'
    observer.publish('Generation', 'Complete development capture independently verified.', result['frames'], result['frames'], state='completed')
    return result


def diagnostic(args, observer):
    selected = contract.decode(contract.read(args.workspace/'material.json', contract.MIB))
    root = Path(selected['capture'])
    config = contract.experiment(contract.decode(contract.read(root/'experiment.json', contract.MIB)))
    entries = {entry['trajectory_id']:entry for entry in trajectories(config)}
    artifacts = args.workspace/'execution/stage-01/artifacts'
    # These are existing independently restored predictions, not new inference.
    with np.load(artifacts/'predictions.npz',allow_pickle=False) as archive:
        pressures = archive['pressure'].copy()
    count = 0
    observer.publish('Teacher-state evaluation', 'Verifying frozen inputs before native diagnostics. No autonomous rollout.', total=len(entries)*2)
    def poll():
        nonlocal count
        journal = args.output/'cases.jsonl'
        if not journal.is_file():
            return
        raw = journal.read_bytes()
        if len(raw)>16*contract.MIB:
            raise ValueError('Diagnostic journal exceeds bound')
        lines = raw.split(b'\n')[:-1]
        if len(lines)>count:
            index = len(lines)-1
            record = contract.decode(lines[index])
            preview = observer.frame(root, entries[record['trajectory_id']],record['frame'],pressures[index],
                                     'Learned pressure proposal (Pa); '+('passes native gates' if record['raw_learned']['qualified'] else 'fails native gates; qualified solve uses fallback'))
            count = len(lines)
            observer.publish('Teacher-state evaluation', f'Case {count}/{len(entries)*2}: raw learned qualified={record["raw_learned"]["qualified"]}; fallback={record["learned_jacobi"]["reset_fallback"]}. Reference tracer shown.', count,len(entries)*2,preview=preview)
    # The native process owner requires one Python thread and no unrelated
    # children. Keep observation in the parent and science in a sole worker.
    command = [sys.executable, str(Path(__file__).with_name('evaluation_worker.py')), '--workspace', str(args.workspace), '--output', str(args.output)]
    with capture.legacy.bounded_process(command, 600, stdin=subprocess.DEVNULL) as worker:
        try:
            while worker.poll() is None:
                poll()
                time.sleep(0.25)
        except BaseException:
            if worker.poll() is None:
                worker.send_signal(signal.SIGINT)
                worker.wait(timeout=65)
            raise
        if worker.returncode != 0:
            raise RuntimeError(f'Native evaluation worker failed with exit {worker.returncode}; inspect preserved output.')
    poll()
    result = contract.decode(contract.read(args.output/'evaluation.json',4*contract.MIB))
    atomic(observer.root/'evaluation.json',result)
    observer.publish('Teacher-state evaluation',f'Complete: raw learned {result["raw_qualified"]}/{result["cases"]}; fallback-qualified {result["methods"]["learned_jacobi"]["qualified"]}/{result["cases"]}. No acceleration established.',result['cases'],result['cases'],state='completed')
    return result


def watch_run(args, observer):
    # Use Tuldok's authenticated loopback client; never read/copy catalog tokens.
    sys.path.insert(0, str(args.tuldok_source))
    from agent_api import AgentClient
    client = AgentClient.from_config(args.agent_config)
    deadline = time.monotonic()+args.seconds
    while time.monotonic()<deadline:
        response = client.call('runs.progress', {'run_id':args.run_id})
        if response['status'] != 200:
            raise RuntimeError('Tuldok Run progress is unavailable: HTTP '+str(response['status']))
        value = response['result']
        progress = value.get('progress') or {}
        done, total = progress.get('completed',0), progress.get('total')
        state = value['state']
        terminal = state in ('completed','failed','cancelled','unknown')
        if state=='completed' and value.get('custody_phase')!='drained':
            raise RuntimeError('Run completion has no drained custody; inspect Tuldok.')
        label = progress.get('stage','Managed Run')
        observer.publish('Training / managed execution', f'Run {args.run_id}: {label}, {state}; catalog {client.catalog_id}.',
                         done,total,state=state if terminal and state!='unknown' else 'failed' if state=='unknown' else 'running')
        if terminal:
            return {'run_id':args.run_id,'state':state,'custody_phase':value.get('custody_phase')}
        time.sleep(1)
    raise RuntimeError('Observation budget elapsed; the Tuldok Run remains managed by Tuldok.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',type=Path,default=LAB/'runs/live-project')
    sub = parser.add_subparsers(dest='command',required=True)
    generate_parser = sub.add_parser('capture')
    for name in ('config','binary','output'):
        generate_parser.add_argument('--'+name,type=Path,required=True)
    evaluation_parser = sub.add_parser('evaluate')
    for name in ('workspace','output'):
        evaluation_parser.add_argument('--'+name,type=Path,required=True)
    training_parser = sub.add_parser('watch-run')
    training_parser.add_argument('--tuldok-source',type=Path,required=True)
    training_parser.add_argument('--agent-config',type=Path,required=True)
    training_parser.add_argument('--run-id',required=True)
    training_parser.add_argument('--seconds',type=int,default=600)
    args = parser.parse_args()
    if args.command=='watch-run' and not 1 <= args.seconds <= 3600:
        parser.error("--seconds must be between 1 and 3600")
    observer = Observer(args.live)
    try:
        result = generate(args,observer) if args.command=='capture' else diagnostic(args,observer) if args.command=='evaluate' else watch_run(args,observer)
        print(json.dumps({key:value for key,value in result.items() if key != "frame_metrics"}))
    except BaseException as error:
        observer.publish(args.command, str(error)[:1000] or type(error).__name__,state='cancelled' if isinstance(error,KeyboardInterrupt) else 'failed')
        raise


if __name__=='__main__':
    main()
