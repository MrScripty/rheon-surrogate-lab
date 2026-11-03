"""Check strict legacy force inputs against native force previews on consumed scenes."""
import argparse
import os
from pathlib import Path
from support import ROOT, read, write, digest
from legacy_exact_forcing import forcing
from native_forcing_client import NativeForcingClient
from recurrent_sampler_native import dimensions
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh strict legacy force qualification required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered legacy force source changed: '+path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed'])
    args.output.mkdir(parents=True)
    rows, owners = [], []
    with torch.inference_mode():
        for case in reg['cases']:
            scene = case['scene']
            client = NativeForcingClient(scene, args.output/case['identifier'])
            try:
                before_packet, before = client.inspect()
                for step in reg['steps']:
                    increments = forcing(scene, step, 1)[0]
                    fixtures = [('zero', [torch.zeros(s, dtype=torch.float32)
                                          for s in dimensions(scene['counts'])[:3]]),
                        ('random', [torch.randn(s, dtype=torch.float32)*.01
                                    for s in dimensions(scene['counts'])[:3]]),
                        ('canceling', [(-v).float() for v in increments])]
                    for kind, base in fixtures:
                        native = client.preview_force(step, [v.numpy() for v in base])
                        merged = [(v.double()+addition).float() for v, addition in zip(base, increments)]
                        # The native external-force stage skips walls, including their zero sign bits.
                        for component, value in enumerate(merged):
                            wall = [slice(None)]*3
                            wall[2-component] = [0, -1]
                            value[tuple(wall)] = base[component][tuple(wall)]
                        if any(a.tobytes() != b.numpy().tobytes() for a, b in zip(native, merged)):
                            raise ValueError('Strict legacy force bytes differ: '+str((case['identifier'], step, kind)))
                        rows.append(dict(identifier=case['identifier'], step=step, base=kind,
                            counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                            complete_merged_velocity_bitwise_equal=True))
                after_packet, after = client.inspect()
                if before_packet != after_packet or any(before[k].tobytes() != after[k].tobytes() for k in before):
                    raise ValueError('Legacy force preview mutated state or clock')
            finally:
                owner = client.close()
                owners.append(owner)
                if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                    raise ValueError('Every force-preview child must complete and reap')
    if len(rows) != reg['expected_comparisons']:
        raise ValueError('Full registered exact-force comparison population required')
    write(args.output/'qualification.json', dict(status='passed_strict_legacy_external_forcing',
        comparisons=rows, native_processes=owners, source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), numerical_steps_executed=0,
        model_targets_verified=False, optimizer_updates=0, reserved_references_consumed=False))
    print('Qualified', len(rows), 'strict legacy external-force comparisons', flush=True)
