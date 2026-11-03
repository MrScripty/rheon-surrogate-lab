"""Compare the callable generator against every frozen diagnostic output, all K."""
import argparse
import os
from pathlib import Path
from support import ROOT, read, write, digest
from divisor_generator import load_model
from diagnose_causal_composition import FIELDS, inventory
from evaluate_recurrent_sampler import identity
import numpy as np
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh callable generator qualification required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Callable generator source changed: '+path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    control = read(ROOT/reg['control'])
    expected = {(r['identifier'], r['step']): r['output_identity'] for r in control['rows']}
    model = load_model(reg['pair'], reg['library_path'])
    before = inventory(model)
    rows = []
    try:
        with torch.inference_mode():
            for trajectory in read(ROOT/reg['bank'])['trajectories']:
                receipt_path = ROOT/trajectory['receipt']
                if digest(receipt_path) != trajectory['receipt_sha256']:
                    raise ValueError('Current physical input receipt changed')
                receipt = read(receipt_path)
                entry = receipt['states'][0]
                source = receipt_path.parent/entry['path']
                if digest(source) != entry['sha256']:
                    raise ValueError('Current physical input changed')
                with np.load(source, allow_pickle=False) as archive:
                    initial = {key: archive[key].copy() for key in FIELDS}
                for k in reg['horizons']:
                    current = initial
                    for start in range(0, 240, k):
                        outputs = model.predict(current, receipt['scene'], start, k)
                        if len(outputs) != k:
                            raise ValueError('Every complete intermediate state required')
                        for lead, output in enumerate(outputs, 1):
                            step = start+lead
                            if identity(output) != expected[(trajectory['identifier'], step)]:
                                raise ValueError('Callable model changed actual complete output bytes')
                            rows.append(dict(identifier=trajectory['identifier'], K=k, step=step,
                                every_physical_field_bitwise_equal=True))
                        current = outputs[-1]
                    print('Qualified callable', trajectory['identifier'], 'K', k, flush=True)
        if inventory(model) != before or torch.cuda.is_initialized():
            raise ValueError('Callable qualification changed restored tensors or device')
    finally:
        model.close()
    if len(rows) != reg['expected_complete_states']:
        raise ValueError('Full callable qualification population required')
    write(args.output/'qualification.json', dict(status='passed_callable_divisor_generator_output_parity',
        rows=rows, source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        every_intermediate_state_scored=True, optimizer_updates=0, latency_claim=False,
        independent_confirmation=False, full_generator_target_pass=False))
