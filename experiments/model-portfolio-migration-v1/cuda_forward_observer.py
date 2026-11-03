"""Scoped forward-device observation with a genuinely hook-free primary mode."""
from collections import Counter
from contextlib import contextmanager


class CUDAForwardObserver:
    def __init__(self,torch):
        self.torch = torch
        self.handle = None
        self.phase = None
        self.calls = Counter()
        self.transitions = Counter()
        self.forbidden_attempts = 0
        self.closed = False

    def _guard(self,module,args):
        tensors = []
        def collect(value):
            if isinstance(value,self.torch.Tensor):
                tensors.append(value)
            elif isinstance(value,(tuple,list)):
                for item in value:
                    collect(item)
            elif isinstance(value,dict):
                for item in value.values():
                    collect(item)
        collect(args)
        local = [*module.parameters(recurse=False),*module.buffers(recurse=False)]
        if not tensors or any(v.device.type != 'cuda' for v in [*tensors,*local]):
            self.forbidden_attempts += 1
            raise ValueError('CPU/unclassified neural execution blocked before forward')
        self.calls[(self.phase,type(module).__name__)] += 1

    def enable(self,phase):
        if self.closed or self.handle is not None:
            raise ValueError('Observer must be open and detached before enabling')
        self.phase = phase
        self.handle = self.torch.nn.modules.module.register_module_forward_pre_hook(self._guard)
        self.transitions[('enabled',phase)] += 1

    def disable(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.transitions[('disabled',self.phase)] += 1

    def set_phase(self,phase):
        if self.handle is None or self.closed:
            raise ValueError('Phase changes require an active observer')
        self.phase = phase

    @contextmanager
    def unobserved(self,phase):
        if self.handle is None or self.closed:
            raise ValueError('An active observer must own each unobserved scope')
        previous = self.phase
        self.disable()
        self.phase = phase
        self.transitions[('unobserved',phase)] += 1
        try:
            yield
        finally:
            self.enable(previous)

    def close(self):
        self.disable()
        self.closed = True

    def evidence(self):
        return dict(forbidden_forward_attempts=self.forbidden_attempts,closed=self.closed,
            observer_attached=self.handle is not None,
            calls_by_phase={phase:{name:count for (p,name),count in self.calls.items() if p == phase}
                for phase in {p for p,_ in self.calls}},
            transitions={action+':'+str(phase):count for (action,phase),count in self.transitions.items()},
            scope='Calls are observed only while attached. Primary/off-calibration calls use identical frozen CUDA code/inputs and exact output hashes, with no observer callback installed.')
