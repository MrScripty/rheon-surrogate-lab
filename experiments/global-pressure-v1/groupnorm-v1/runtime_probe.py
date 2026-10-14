#!/usr/bin/env python3
"""Neutral captured-runtime GN arithmetic; no optimizer, grant or scientific Run."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    output = Path(sys.argv[1])
    if output.exists() or output.is_symlink():
        raise ValueError("fresh runtime probe report required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    group = torch.nn.GroupNorm(4, 8, eps=1e-5, affine=False).double()
    x = torch.linspace(-2, 3, 128, dtype=torch.float64).reshape(1, 8, 2, 2, 4).requires_grad_()
    actual = group(x)
    grouped = x.reshape(1, 4, -1)
    expected = ((grouped-grouped.mean(-1, keepdim=True)) /
                torch.sqrt(grouped.var(-1, unbiased=False, keepdim=True)+1e-5)).reshape_as(x)
    error = float((actual-expected).detach().abs().max())
    assert error < 2e-14 and not list(group.parameters()) and not list(group.buffers())
    objective = (actual*torch.linspace(.1, 2, 128, dtype=torch.float64).reshape_as(x)).sum()
    derivative, = torch.autograd.grad(objective, x)
    assert torch.isfinite(derivative).all() and float(derivative.abs().max()) > 0
    mixed = group(torch.cat((x.detach(), x.detach()*100+7), 0))[:1]
    assert torch.equal(actual.detach(), mixed)
    group.eval()
    assert torch.equal(actual.detach(), group(x.detach()))
    root = Path(np.__file__).parents[1]
    snapshot = root.parent
    mappings = sorted({line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
                       if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[5].startswith('/')})
    assert all(Path(path).is_relative_to(snapshot) for path in mappings)
    assert torch.version.cuda is None and torch.__version__ == '2.8.0+cpu' and np.__version__ == '2.3.5'
    value = {"format": "rheon-groupnorm-runtime-probe-v1", "status": "passed",
             "scope": "direct pinned production captured-runtime bootstrap; neutral arithmetic; no grant/Run/training",
             "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
             "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
             "groupnorm_manual_max_abs_error": error,
             "gradient_max_abs": float(derivative.abs().max()),
             "batch_neighbor_exact": True, "train_eval_exact": True,
             "parameter_count": 0, "buffer_count": 0, "native_mappings": mappings,
             "snapshot": snapshot.name,
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({"status": "passed", "manual_error": error, "report": str(output)}))


if __name__ == '__main__':
    main()
