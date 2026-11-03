"""Finite real CPU tensor workload for device/lifecycle qualification only."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=int, required=True)
    args = parser.parse_args()
    assert 1 <= args.seconds <= 180
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    values = torch.linspace(-1, 1, 64 ** 3, dtype=torch.float32).reshape(64, 64, 64)
    started = time.time()
    marker = {'device': 'cpu', 'cuda_initialized': False, 'compute_started_unix': started,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'optimizer_updates': 0}
    (args.output_dir / 'started.json').write_text(json.dumps(marker))
    deadline = time.monotonic() + args.seconds
    iterations = 0
    while time.monotonic() < deadline:
        values = torch.sin(values) * .99 + .001
        iterations += 1
    assert torch.isfinite(values).all() and not torch.cuda.is_initialized()
    result = {**marker, 'compute_finished_unix': time.time(), 'iterations': iterations,
        'checksum': float(values.sum()), 'torch_threads': torch.get_num_threads(),
        'scope': 'real CPU arithmetic and overlap qualification; no scientific/model accuracy claim'}
    (args.output_dir / 'receipt.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main()
