"""Exercise the worker's actual calibration statement without model execution.

The seam is Python cache-policy control flow. Only the timing callback and
observer transitions are substituted; the loop body comes from the real worker.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
import unittest


class Observer:
    def set_phase(self,phase):
        pass

    @contextmanager
    def unobserved(self,phase):
        yield


class SharedCacheModeTests(unittest.TestCase):
    def test_calibration_preserves_cache_policy_and_archive_branch(self):
        source = Path(__file__).with_name('shared_component_benchmark.py')
        run = next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name == 'run')
        loops = [node for node in ast.walk(run) if isinstance(node,ast.For) and isinstance(node.target,ast.Name)
                 and node.target.id == 'number' and 'calibration_pairs_per_case' in ast.unparse(node.iter)]
        self.assertEqual(len(loops),1)
        # Executing this real statement reproduces the local-name assignment
        # semantics that determine the archive branch after all timed calls.
        body = ast.fix_missing_locations(ast.Module(body=loops,type_ignores=[]))
        program = compile(body,str(source),'exec')
        for mode in ('unprepared','prepared'):
            for phase in (0,1):
                with self.subTest(mode=mode,phase=phase):
                    calls = []
                    def sample():
                        calls.append(1)
                        return 100
                    values = dict(mode=mode,phase=phase,index=0,protocol=dict(calibration_pairs_per_case=4),
                        observer=Observer(),sample=sample,pairs=[])
                    exec(program,values)
                    self.assertEqual(values['mode'],mode)
                    self.assertEqual(len(calls),8)
                    self.assertEqual(len(values['pairs']),4)
                    self.assertEqual(phase == 0 and values['mode'] == 'unprepared',phase == 0 and mode == 'unprepared')


if __name__ == '__main__':
    unittest.main()
