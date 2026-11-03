"""Bounded observational Adam archive; no optimizer or RNG ownership."""
import io
import json
import time
import zipfile

import numpy as np
import torch

UPDATES = (1, 10, 100, 150, 200)
CAP = 48 * 1024**2
ACTIVE = None


class Trace:
    def __init__(self, path, network):
        self.paths = {u:path.with_name('adam-trace-%03d.npz'%u) for u in UPDATES}
        self.archives = {u:zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED) for u,p in self.paths.items()}
        self.layout = [dict(name=n, shape=list(p.shape), count=p.numel())
                       for n, p in network.named_parameters()]
        self.count = sum(row['count'] for row in self.layout)
        if self.count * 4 * 6 * len(UPDATES) + 1024**2 > CAP:
            [archive.close() for archive in self.archives.values()]
            raise ValueError('Predeclared vector and metadata budget exceeded')
        self.records = []
        self.pending = None
        self.elapsed_s = 0.
        self.payload_bytes = 0

    def array(self, name, values):
        # One host vector and one serialization buffer, never an update history on GPU.
        array = np.concatenate([v.detach().reshape(-1).cpu().numpy() for v in values])
        if array.dtype != np.float32 or array.size != self.count or not np.isfinite(array).all():
            raise ValueError('Finite float32 vector in captured parameter order required')
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        self.payload_bytes += buffer.tell()
        if self.payload_bytes > CAP - 1024**2:
            raise ValueError('Diagnostic archive vector budget exceeded')
        archive = self.archives[self.pending['update']]
        archive.writestr(name + '.npy', buffer.getvalue())

    def vector(self, label, values):
        self.array('u%03d_%s' % (self.pending['update'], label), values)

    def finish(self):
        if self.pending is not None or [r['update'] for r in self.records] != list(UPDATES):
            raise ValueError('Complete predeclared update capture required')
        document = dict(schema='rheon-adam-observation-v1', layout=self.layout,
            parameter_count=self.count, updates=list(UPDATES), records=self.records,
            observer_elapsed_s=self.elapsed_s, vector_payload_bytes=self.payload_bytes,
            vector_names=['parameters_before', 'parameters_after', 'exp_avg_before',
                          'exp_avg_sq_before', 'gradient_raw', 'gradient_clipped'],
            delta_definition='parameters_after minus parameters_before in float64 analysis',
            diagnostic_network_forwards=len(UPDATES)*2*8*12,
            scope='Observed unchanged aggregate training; diagnostics excluded from optimizer forward counters.')
        raw = json.dumps(document, sort_keys=True, allow_nan=False).encode()
        if len(raw) > 1024**2:
            raise ValueError('Diagnostic metadata budget exceeded')
        for archive in self.archives.values():
            archive.writestr('metadata.npy', npy(np.frombuffer(raw, dtype=np.uint8)))
            archive.close()
        if any(p.stat().st_size > 8*1024**2 for p in self.paths.values()):
            raise ValueError('Compressed trace exceeds evidence shard limit')
        return document


def npy(value):
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def start(path, network):
    global ACTIVE
    if ACTIVE is not None:
        raise ValueError('One observer per trainer process')
    ACTIVE = Trace(path, network)


def metrics(network, normalization):
    import original_fit_bank as bank
    training = network.training
    try:
        network.eval()
        with torch.no_grad():
            losses = [float(bank.loss(network, sample, normalization)) for sample in bank.SAMPLES]
            pressure = bank.measure(network, normalization, bank.SAMPLES)
        return dict(losses=losses, pressure=pressure)
    finally:
        network.train(training)


def before_update(network, optimizer, normalization, update):
    trace = ACTIVE
    if trace is None or update not in UPDATES:
        return
    started = time.perf_counter()
    if trace.pending is not None or len(optimizer.param_groups) != 1:
        raise ValueError('Sequential single-group Adam capture required')
    parameters = list(network.parameters())
    states = [optimizer.state.get(p, {}) for p in parameters]
    row = dict(update=update, group={k:v for k,v in optimizer.param_groups[0].items() if k != 'params'},
        initialized_before=[bool(s) for s in states],
        steps_before=[float(s['step']) if s else 0. for s in states],
        before=metrics(network, normalization))
    trace.pending = row
    trace.vector('parameters_before', parameters)
    for key in ('exp_avg', 'exp_avg_sq'):
        # Explicit Adam initialization rule, not reconstructed historical moments.
        trace.vector(key+'_before', [s[key] if s else torch.zeros_like(p) for p,s in zip(parameters,states,strict=True)])
    trace.elapsed_s += time.perf_counter()-started


def gradients(network, label):
    trace = ACTIVE
    if trace is None or trace.pending is None:
        return
    started = time.perf_counter()
    trace.vector(label, [p.grad for p in network.parameters()])
    trace.elapsed_s += time.perf_counter()-started


def after_update(network, optimizer, normalization, update):
    trace = ACTIVE
    if trace is None or update not in UPDATES:
        return
    started = time.perf_counter()
    if trace.pending is None or trace.pending['update'] != update:
        raise ValueError('Matching pre-update state required')
    trace.vector('parameters_after', list(network.parameters()))
    trace.pending.update(after=metrics(network, normalization),
        steps_after=[float(optimizer.state[p]['step']) for p in network.parameters()])
    trace.records.append(trace.pending)
    trace.pending = None
    trace.elapsed_s += time.perf_counter()-started


def finish():
    global ACTIVE
    if ACTIVE is None:
        raise ValueError('Explicit observation session required')
    result = ACTIVE.finish()
    ACTIVE = None
    return result
