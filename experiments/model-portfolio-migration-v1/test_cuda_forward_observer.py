"""Observer ownership checks without importing a neural runtime."""
from types import SimpleNamespace
import unittest

from cuda_forward_observer import CUDAForwardObserver


class Tensor:
    def __init__(self,device='cuda'):
        self.device = SimpleNamespace(type=device)


class Module:
    def __init__(self,parameter_device='cuda'):
        self.parameter = Tensor(parameter_device)
        self.completed_forwards = 0

    def parameters(self,recurse):
        return [self.parameter]

    def buffers(self,recurse):
        return []


class Registry:
    def __init__(self):
        self.hooks = {}
        self.counter = 0

    def register_module_forward_pre_hook(self,hook):
        key = self.counter
        self.counter += 1
        self.hooks[key] = hook
        return SimpleNamespace(remove=lambda:self.hooks.pop(key))

    def forward(self,module,argument):
        for hook in list(self.hooks.values()):
            hook(module,(argument,))
        module.completed_forwards += 1


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()
        torch = SimpleNamespace(Tensor=Tensor,nn=SimpleNamespace(modules=SimpleNamespace(module=self.registry)))
        self.observer = CUDAForwardObserver(torch)
        self.module = Module()
        self.addCleanup(self.observer.close)

    def test_primary_removes_callback_and_restores_phase(self):
        self.observer.enable('warmup')
        self.registry.forward(self.module,Tensor())
        before = dict(self.observer.calls)
        with self.observer.unobserved('primary'):
            self.assertFalse(self.registry.hooks)
            self.registry.forward(self.module,Tensor())
            self.assertEqual(before,dict(self.observer.calls))
        self.assertEqual(len(self.registry.hooks),1)
        self.assertEqual(self.observer.phase,'warmup')
        self.registry.forward(self.module,Tensor())
        self.assertEqual(self.observer.calls[('warmup','Module')],2)

    def test_exception_reattaches_before_propagating(self):
        self.observer.enable('setup')
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with self.observer.unobserved('primary'):
                raise RuntimeError('fixture')
        self.assertEqual(len(self.registry.hooks),1)
        self.assertEqual(self.observer.phase,'setup')

    def test_cpu_input_and_local_parameter_block_before_forward(self):
        self.observer.enable('construction')
        with self.assertRaisesRegex(ValueError,'before forward'):
            self.registry.forward(self.module,{'nested':[Tensor('cpu')]})
        self.assertEqual(self.module.completed_forwards,0)
        cpu_parameter = Module('cpu')
        with self.assertRaisesRegex(ValueError,'before forward'):
            self.registry.forward(cpu_parameter,Tensor())
        self.assertEqual(cpu_parameter.completed_forwards,0)
        self.assertEqual(self.observer.forbidden_attempts,2)

    def test_nested_unobserved_and_closed_reenable_rejected(self):
        self.observer.enable('warmup')
        with self.observer.unobserved('primary'):
            with self.assertRaisesRegex(ValueError,'active observer'):
                with self.observer.unobserved('nested'):
                    self.fail('nested scope admitted')
        self.observer.close()
        self.observer.close()
        self.assertFalse(self.registry.hooks)
        self.assertTrue(self.observer.evidence()['closed'])
        with self.assertRaisesRegex(ValueError,'open and detached'):
            self.observer.enable('late')


if __name__ == '__main__':
    unittest.main()
