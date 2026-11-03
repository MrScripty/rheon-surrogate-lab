"""Bounded reusable CUDA transfer staging with independently owned CPU results.

Construct once for one native layout and at most four output states. Each run
is a synchronous transaction on the construction stream. The callback borrows
the internal CUDA input only until it returns; it must enqueue all work on that
stream. Returned CPU states own fresh ordinary storage and survive later runs.
No model, force, numerical operator or physical computation lives here.
"""
import threading
import time
import torch
from cuda_native_transfer_v1 import FIELDS, layout


class NativeStateWorkspace:
    def __init__(self, fields, maximum_horizons=4):
        if maximum_horizons not in (1, 2, 3, 4):
            raise ValueError('One to four complete output states required')
        started = time.perf_counter()
        entries, size, gaps = layout([fields], 'cpu')
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.stream = torch.cuda.current_stream(self.device)
        self.lock = threading.Lock()
        self.maximum_horizons = maximum_horizons
        self.input_entries = entries
        self.pinned_input = torch.empty(size, dtype=torch.uint8, device='cpu', pin_memory=True)
        self.device_input = torch.empty(size, dtype=torch.uint8, device=self.device)
        for offset, length in gaps:
            self.pinned_input.narrow(0, offset, length).zero_()
        self.input_sources = {n: self.pinned_input.narrow(0, offset, length)
                              for _, n, offset, length, _, _ in entries}
        self.borrowed_input = {n: self.device_input.narrow(0, offset, length).view(dtype).reshape(shape)
                               for _, n, offset, length, shape, dtype in entries}
        _, output_size, output_gaps = layout([self.borrowed_input] * maximum_horizons, 'cuda')
        self.pinned_output = torch.empty(output_size, dtype=torch.uint8, device='cpu', pin_memory=True)
        for offset, length in output_gaps:
            self.pinned_output.narrow(0, offset, length).zero_()
        self.output_layouts = {}
        self.output_targets = {}
        for k in range(1, maximum_horizons + 1):
            specs, used, _ = layout([self.borrowed_input] * k, 'cuda')
            self.output_layouts[k] = (specs, used)
            self.output_targets[k] = [self.pinned_output.narrow(0, offset, length).view(dtype).reshape(shape)
                                      for _, _, offset, length, shape, dtype in specs]
        self.setup = dict(maximum_horizons=maximum_horizons,
                          pinned_bytes=size + output_size, gpu_input_bytes=size,
                          setup_wall_ms=1000 * (time.perf_counter() - started),
                          output_ownership='Fresh non-pinned CPU allocation for each complete call',
                          concurrency='One synchronous transaction on the construction CUDA stream')

    def _validate(self, fields):
        if torch.cuda.current_device() != self.device.index or torch.cuda.current_stream(self.device) != self.stream:
            raise ValueError('Workspace must run on its construction CUDA device and stream')
        if set(fields) != set(FIELDS):
            raise ValueError('Complete native fields required')
        for _, name, _, _, shape, dtype in self.input_entries:
            value = fields[name]
            if value.device.type != 'cpu' or value.dtype != dtype or value.shape != shape or not value.is_contiguous():
                raise ValueError('Exact prepared CPU native field layout required')

    @torch.inference_mode()
    def run(self, fields, generate):
        """CPU native state -> callback on borrowed CUDA state -> owned CPU states.

        Packing, upload, callback, download, synchronization, ordinary CPU
        allocation/copy and all layout checks are part of this complete call.
        On callback failure the stream is drained before staging can be reused.
        """
        self._validate(fields)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Workspace transaction already active')
        queued = False
        try:
            for name, target in self.input_sources.items():
                target.copy_(fields[name].reshape(-1).view(torch.uint8))
            self.device_input.copy_(self.pinned_input, non_blocking=True)
            queued = True
            outputs = generate(self.borrowed_input)
            k = len(outputs)
            if k not in self.output_layouts:
                raise ValueError('Prepared one-to-four complete outputs required')
            entries, size, _ = layout(outputs, 'cuda')
            expected, expected_size = self.output_layouts[k]
            if entries != expected or size != expected_size:
                raise ValueError('Generated fields do not match prepared native output layout')
            if any(v.device != self.device for row in outputs for v in row.values()):
                raise ValueError('Generated fields must use the construction CUDA device')
            for spec, target in zip(entries, self.output_targets[k]):
                step, name = spec[:2]
                target.copy_(outputs[step][name], non_blocking=True)
            self.stream.synchronize()
            queued = False
            # Explicitly request ordinary storage: cloning a pinned tensor does
            # not provide a sufficiently clear allocation/ownership contract.
            owned = torch.empty(size, dtype=torch.uint8, device='cpu', pin_memory=False)
            owned.copy_(self.pinned_output.narrow(0, 0, size))
            result = [{} for _ in outputs]
            for step, name, offset, length, shape, dtype in entries:
                result[step][name] = owned.narrow(0, offset, length).view(dtype).reshape(shape)
            return result
        finally:
            try:
                if queued:
                    self.stream.synchronize()
            finally:
                self.lock.release()
