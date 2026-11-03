"""Independent event-order checks; fake calls execute no neural model."""
from types import SimpleNamespace
import unittest

from complete_generator_gpu_benchmark import sample


class TimingBoundary(unittest.TestCase):
    def fixture(self):
        events = []
        hooks = {}
        module = SimpleNamespace(_global_forward_pre_hooks=hooks,_global_forward_hooks={})
        def register(hook):
            events.append('install')
            hooks[0] = hook
            def remove():
                events.append('remove')
                hooks.clear()
            return SimpleNamespace(remove=remove)
        module.register_module_forward_pre_hook = register
        torch = SimpleNamespace(nn=SimpleNamespace(modules=SimpleNamespace(module=module)),
            cuda=SimpleNamespace(synchronize=lambda:events.append('sync')))
        ticks = iter([1000000,4000000])
        def clock():
            events.append('clock')
            return next(ticks)
        return events,hooks,torch,clock

    def test_primary_has_no_instrumentation(self):
        events,hooks,torch,clock = self.fixture()
        def call():
            self.assertFalse(hooks)
            events.append('predict')
            return 'allKownedCPUstates'
        self.assertEqual(sample(call,torch,clock=clock),(3.,'allKownedCPUstates'))
        self.assertEqual(events,['sync','clock','predict','sync','clock'])

    def test_calibration_hook_setup_and_cleanup_outside_clock(self):
        events,hooks,torch,clock = self.fixture()
        hook = object()
        def call():
            self.assertIs(hooks[0],hook)
            events.append('predict')
        sample(call,torch,hook,clock)
        self.assertEqual(events,['install','sync','clock','predict','sync','clock','remove'])
        self.assertFalse(hooks)

    def test_failed_prediction_removes_observer(self):
        events,hooks,torch,clock = self.fixture()
        def call():
            raise ValueError('specific prediction failure')
        with self.assertRaisesRegex(ValueError,'^specific prediction failure$'):
            sample(call,torch,object(),clock)
        self.assertFalse(hooks)
        self.assertEqual(events,['install','sync','clock','remove'])

    def test_ambient_instrumentation_is_rejected_before_prediction(self):
        events,hooks,torch,clock = self.fixture()
        hooks[4] = object()
        with self.assertRaisesRegex(ValueError,'Unexpected global forward instrumentation'):
            sample(lambda:self.fail('Must not predict'),torch,clock=clock)
        self.assertEqual(events,[])


if __name__ == '__main__':
    unittest.main()
