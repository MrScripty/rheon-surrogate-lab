import unittest
import numpy as np
import final_spectral as spectral


class AnalyticBaselineContracts(unittest.TestCase):
    def test_hand_chain_and_constant_nullspace(self):
        basis = spectral.cosine_basis([3, 1, 1])
        values = spectral.eigenvalues([3, 1, 1], [1., 1., 1.], 1.)
        np.testing.assert_allclose(values, [1., 3.], atol=1e-14)
        result = spectral.solve(np.array([[-4., 6., -2.]]), basis, values)
        np.testing.assert_allclose(result, [[0., 4., 2.]], atol=1e-14)
        np.testing.assert_array_equal(spectral.solve(np.zeros((1,3)), basis, values), 0.)

    def test_fixed_operator_modes_against_public_rheon_export(self):
        # This export was generated on geometry only before the candidate run;
        # no reserved seed or final target is read. Skip when it is absent in CI.
        from pathlib import Path
        path = Path('runs/global-v1/operator/operator.bin')
        if not path.exists():
            self.skipTest('run development operator export to check pinned action')
        matrix = np.fromfile(path, dtype='<f8').reshape(336,336)
        basis = spectral.cosine_basis([8,7,6])
        values = spectral.eigenvalues([8,7,6], [.25,.3,.4], 1.25)
        np.testing.assert_allclose(matrix @ basis, basis * values, atol=5e-15, rtol=5e-13)
        rhs = np.sin(np.arange(336) * .37)[None,:]
        rhs -= rhs.mean(axis=1, keepdims=True)  # Synthetic unit fixture only.
        pressure = spectral.solve(rhs, basis, values)
        np.testing.assert_allclose(pressure @ matrix.T, rhs, atol=1e-13)

class FrozenProtocolContracts(unittest.TestCase):
    def test_balanced_order_and_distinct_test_filter(self):
        import collections
        import final_evaluate as final
        rows = final.orders()
        self.assertEqual(len(rows), 12)
        for position in range(6):
            self.assertEqual(collections.Counter(row[position] for row in rows), {m:2 for m in final.METHODS})
        pairs = collections.Counter((a,b) for row in rows for a,b in zip(row,row[1:]))
        self.assertEqual(pairs, {(a,b):2 for a in final.METHODS for b in final.METHODS if a != b})
        import train_cpu as common
        from pathlib import Path
        # Filtering precedes all reads; nonexistent final arrays cannot be opened.
        self.assertEqual(common.load_cases(Path('/nonexistent'), {'cases':[{'spec':{'split':'final_test'}}]}), [])

    def test_failed_projection_is_reported_without_promoting_acceptance(self):
        import final_evaluate as final
        summary = final.summarize([{'id':'fixture','qualified':False,'solver_error':'iteration limit'}])
        self.assertEqual(summary['qualified'], 0)
        self.assertEqual(summary['failed_ids'], ['fixture'])
        self.assertIsNone(summary['mean_reference_relative_rmse'])
        self.assertEqual(summary['solver_errors'], {'fixture':'iteration limit'})

class FreezeIntegrityContracts(unittest.TestCase):
    def test_source_binary_and_checkpoint_mutations_are_rejected(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from contextlib import ExitStack
        import final_evaluate as final
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source, binary, weights = (root / name for name in ('source.py', 'probe', 'weights.npz'))
            for path in (source, binary, weights):
                path.write_bytes(b'original')
            identity = {'fixture':'native'}
            frozen = {'schema':'rheon-final-freeze-v1','status':'frozen_before_final_generation',
                      'limits':final.LIMITS,'methods':list(final.METHODS),'method_orders':final.orders(),
                      'seeds':list(range(9000,9016)),'runtime':{},
                      'source_sha256':{'source.py':final.digest(source)},
                      'binaries':{'probe':final.digest(binary)},'candidate_weights_sha256':final.digest(weights),
                      'reviewed_patch_tree':'tree','native_identity':identity}
            freeze = root/'freeze.json'
            freeze.write_text(json.dumps(frozen))
            for name,value in [('ROOT',root),('WEIGHTS',weights),('PROBE',binary)]:
                stack.enter_context(patch.object(final,name,value))
            stack.enter_context(patch.object(final,'require_clean'))
            stack.enter_context(patch.object(final,'git',return_value='tree'))
            stack.enter_context(patch.object(final,'runtime',return_value={}))
            stack.enter_context(patch.object(final,'source_files',return_value=['source.py']))
            stack.enter_context(patch.object(final.subprocess,'check_output',return_value=json.dumps(identity)))
            self.assertEqual(final.validate_freeze(freeze), frozen)
            for path in (source,binary,weights):
                path.write_bytes(b'mutated')
                with self.assertRaises(ValueError):
                    final.validate_freeze(freeze)
                path.write_bytes(b'original')
