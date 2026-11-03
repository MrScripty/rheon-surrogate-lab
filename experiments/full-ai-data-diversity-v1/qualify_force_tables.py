"""Check reused external-input scalars against the qualified strict implementation."""
import argparse
import os
from pathlib import Path
from support import ROOT, read, write, digest
from continuous_force_program_v2 import sample_scene
from legacy_exact_forcing import schedule
from scheduled_force_fine_controls import force as strict
from scheduled_force_tables import force as tables
from recurrent_sampler_native import dimensions
import json
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh force-table qualification required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered force-table source changed: '+path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed'])
    args.output.mkdir(parents=True)
    rows = []
    with torch.inference_mode():
        for case in reg['cases']:
            scene = case['scene']
            if case['kind'] == 'legacy':
                scene = schedule(json.dumps(scene, sort_keys=True, separators=(',', ':')))
            else:
                scene = sample_scene(scene, case['reference_substeps'])
            for step in case['steps']:
                reference_force, _ = strict(scene, step)
                for kind in ('zero', 'random', 'canceling'):
                    shapes = dimensions(scene['counts'])[:3]
                    base = ([torch.zeros(s, dtype=torch.float32) for s in shapes] if kind == 'zero' else
                            [torch.randn(s, dtype=torch.float32)*.01 for s in shapes] if kind == 'random' else
                            [(-v).float() for v in reference_force])
                    sf, sm = strict(scene, step, base)
                    tf, tm = tables(scene, step, base)
                    if any(a.numpy().tobytes() != b.numpy().tobytes() for a, b in zip((*sf, *sm), (*tf, *tm))):
                        raise ValueError('Double force or merged field table parity failed: '+str((case['identifier'], step, kind)))
                    rows.append(dict(identifier=case['identifier'], kind=case['kind'],
                        counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                        substeps=scene.get('reference_substeps', 1), native_step=step, base=kind,
                        double_forces_and_merged_fields_bitwise_equal=True))
    if len(rows) != reg['expected_comparisons'] or torch.cuda.is_initialized():
        raise ValueError('Full force-table qualification population and CPU execution required')
    write(args.output/'qualification.json', dict(status='passed_known_force_table_bitwise_parity',
        comparisons=rows, source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), optimizer_updates=0,
        model_targets_verified=False, latency_claim=False,
        physical_state_evolution=False, numerical_pressure_solver=False))
    print('Qualified', len(rows), 'bitwise double-force and merged-field comparisons', flush=True)
