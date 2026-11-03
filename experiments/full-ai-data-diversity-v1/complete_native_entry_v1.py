"""Synchronous complete native entry with graph output borrowing kept private.

All returned CPU fields own fresh storage. The normal CUDA-returning prediction
still clones its outputs. Every graph mutation is exclusive on the construction
stream, including setup, replay, download, failure recovery and cleanup.
"""
from contextlib import contextmanager
import threading
import torch
from debug.gather_fused_graph_v1 import GatherFusedGraph
from cuda_native_transfer_v1 import FIELDS, layout, upload_state


class CompleteNativeEntry:
    generator_class = GatherFusedGraph

    def __init__(self, pair, shared_pair, build, control_build):
        self._device = torch.cuda.current_device()
        self._stream = torch.cuda.current_stream().cuda_stream
        self._lock = threading.Lock()
        self._closed = False
        self._generator = self.generator_class(pair, shared_pair, build, control_build)
        self.asset_identity = dict(self._generator.asset_identity,
            complete_native_entry='Exclusive synchronous upload/replay/download transaction',
            graph_outputs='Borrowed privately until all K CPU output copies complete; no GPU output clones in native entry',
            CPU_output_ownership='Fresh independent packed pinned allocation for each call and nonoverlapping fields/horizons',
            normal_GPU_predict_ownership='Unchanged independently owned GPU output clones',
            deterministic_settings_changed=False, learned_arithmetic_unchanged=True)

    @contextmanager
    def _exclusive(self, allow_closed=False):
        if (torch.cuda.current_device() != self._device
                or torch.cuda.current_stream().cuda_stream != self._stream):
            raise ValueError('Native entry requires its construction device and stream')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent or reentrant native entry is not supported')
        try:
            if self._closed and not allow_closed:
                raise ValueError('Native entry is closed')
            yield
        finally:
            self._lock.release()

    @property
    def setup_observations(self):
        return self._generator.setup_observations

    def state_dict(self):
        return self._generator.state_dict()

    def prepare_geometry(self, counts):
        with self._exclusive():
            self._generator.prepare_geometry(counts)

    def prepare_block(self, fields, scene, step, horizons, substeps=1):
        with self._exclusive():
            self._generator.prepare_block(fields, scene, step, horizons, substeps)

    def prepare_native(self, fields, scene, step, horizons, substeps=1):
        """Explicit untimed setup from complete current CPU native fields."""
        with self._exclusive():
            uploaded = None
            try:
                uploaded = upload_state(fields)
                self._generator.prepare_geometry(tuple(scene['counts']))
                self._generator.prepare_block(uploaded, scene, step, horizons, substeps)
            finally:
                # Keep owned upload storage alive until queued work has drained.
                torch.cuda.current_stream().synchronize()

    def predict(self, fields, scene, step, horizons, substeps=1):
        with self._exclusive():
            return self._generator.predict(fields, scene, step, horizons, substeps)

    def _replay_borrowed(self, fields, scene, step, horizons, substeps, known):
        """Caller owns the lock, stream and known/input lifetimes until sync."""
        key = tuple(scene['counts']), scene['cells_per_metre'], horizons, substeps
        generator = self._generator
        graph, state, packet, outputs = generator.graphs[key]
        if known.shape != packet.shape:
            raise ValueError('Complete fixed-shape known control packet required')
        generator.pending_packets = [e for e in generator.pending_packets if not e[0].query()]
        if len(generator.pending_packets) >= 64:
            generator.pending_packets[0][0].synchronize()
            generator.pending_packets = [e for e in generator.pending_packets if not e[0].query()]
        for name, target in state.items():
            source = fields[name]
            if (source.device != target.device or source.dtype != target.dtype
                    or source.shape != target.shape or not source.is_contiguous()):
                raise ValueError('Exact native current CUDA state layout required')
            target.copy_(source)
        packet.copy_(known, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        generator.pending_packets.append((ready, known))
        graph.replay()
        return outputs

    @staticmethod
    def _copy_download(outputs, entries, pinned, gaps):
        for offset, length in gaps:
            pinned.narrow(0, offset, length).zero_()
        result = [{} for _ in outputs]
        for step, name, offset, length, shape, dtype in entries:
            target = pinned.narrow(0, offset, length).view(dtype).reshape(shape)
            target.copy_(outputs[step][name], non_blocking=True)
            result[step][name] = target
        return result

    @torch.inference_mode()
    def predict_native(self, fields, scene, step, horizons, substeps=1):
        """Current CPU native fields -> every K independently owned CPU state."""
        with self._exclusive():
            key = tuple(scene['counts']), scene['cells_per_metre'], horizons, substeps
            if (key not in self._generator.graphs or set(fields) != set(FIELDS)
                    or scene.get('sgs', False)):
                raise ValueError('Prepared complete native state required')
            state = self._generator.graphs[key][1]
            for name in FIELDS:
                source, target = fields[name], state[name]
                if (not isinstance(source, torch.Tensor) or source.device.type != 'cpu'
                        or source.dtype != target.dtype or source.shape != target.shape
                        or not source.is_contiguous()):
                    raise ValueError('Exact contiguous CPU native layout required')
            uploaded = known = borrowed = pinned = None
            drained = False
            try:
                uploaded = upload_state(fields)
                known = self._generator.controls.parameters(scene, step, horizons, substeps).pin_memory()
                borrowed = self._replay_borrowed(uploaded, scene, step, horizons, substeps, known)
                entries, size, gaps = layout(borrowed, 'cuda')
                pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                result = self._copy_download(borrowed, entries, pinned, gaps)
                torch.cuda.current_stream().synchronize()
                drained = True
                return result
            finally:
                if not drained:
                    # Locals retain all input and control storage while queued
                    # operations finish, even if replay or a copy raised.
                    torch.cuda.current_stream().synchronize()

    def release_graphs(self):
        with self._exclusive():
            self._generator.release_graphs()

    def close(self):
        with self._exclusive(allow_closed=True):
            if not self._closed:
                self._generator.close()
                self._closed = True
