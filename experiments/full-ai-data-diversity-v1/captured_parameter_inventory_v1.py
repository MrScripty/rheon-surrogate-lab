"""Exact parameter identity without importing diagnostics or numerical solvers."""
import hashlib


def inventory(model):
    return {name: dict(dtype=value.detach().cpu().numpy().dtype.str,
                      shape=list(value.shape),
                      sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest())
            for name, value in model.state_dict().items()}
