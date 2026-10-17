"""Qualification must reject cohort omissions and even tiny numerical regressions."""
import copy
import unittest
import json
from pathlib import Path
import tempfile
import qualify
import qualify_cohort
import dependency_runtime


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.archive={'calls_sha256':'fixed','mean_calls':2.,'maximum_calls':3,
                      'worst_paired_pressure_relative_l2':1e-6}
        self.old=copy.deepcopy(self.archive)
        self.new=copy.deepcopy(self.archive)

    def test_exact_old_reproduction(self):
        self.assertTrue(qualify.accept_case(self.archive,self.old,self.new))
        for key,value in [('calls_sha256','different'),('mean_calls',2.0000000001),
                          ('worst_paired_pressure_relative_l2',1.00000000001e-6)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                qualify.accept_case(self.archive,{**self.old,key:value},self.new)

    def test_no_new_pressure_or_call_allowance(self):
        for key,value in [('mean_calls',2.0000000001),('maximum_calls',4),
                          ('worst_paired_pressure_relative_l2',1.00000000001e-6)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                qualify.accept_case(self.archive,self.old,{**self.new,key:value})

    def test_missing_duplicate_reordered_cases_rejected(self):
        b={'regression':{'cases':[]}}
        for rows in [[],[{'scene':i} for i in range(73)],
                     [{'scene':i} for i in [*range(73),72]],
                     [{'scene':i} for i in reversed(range(74))]]:
            with self.subTest(length=len(rows)),self.assertRaises(ValueError):
                qualify.compare_cohorts(b,rows)

    def test_migration_requires_matched_native_controls(self):
        cases=[]
        for i in range(74):
            group='original58' if i<58 else 'late-amended8' if i<66 else 'minimum8-failed8'
            cases.append({**self.archive,'scene':i,'group':group,'calls':[1,3],
                          'qualified':True,'complete_window':True})
        bindings={'regression':{'cases':copy.deepcopy(cases),
                               'groups':dict.fromkeys(('original58','late-amended8','minimum8-failed8'))}}
        with self.assertRaises(ValueError):
            qualify.compare_cohorts(bindings,cases,copy.deepcopy(cases))
        native=[{'scene':i,'qualified':True,'complete_window':True} for i in range(74)]
        self.assertTrue(qualify.compare_cohorts(bindings,cases,copy.deepcopy(cases),native)['passed'])
        newer=copy.deepcopy(cases)
        newer[0]['worst_paired_pressure_relative_l2']+=1e-16
        newer[1]['worst_paired_pressure_relative_l2']=0.
        with self.assertRaises(ValueError):
            qualify.compare_cohorts(bindings,cases,newer,native)

    def test_native_control_has_no_ai_quality_during_native_prefix(self):
        scene={'steps':16,'warmup_steps':15}
        receipt={'scene':scene,'binary_sha256':'fixed','status':'completed',
                 'native_process':{'reaped':True,'returncode':0},
                 'result':{'reference_control':True,'hard_stop':False},
                 'cpu_threads':1,'cuda_initialized':False}
        frames=[{'step':i,'reference_quality':{'qualified':True},
                 'ai_quality':{'qualified':True} if i==16 else None} for i in range(1,17)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'receipt.json').write_text(json.dumps(receipt))
            (root/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
            self.assertTrue(qualify.native_metrics(root,scene,'fixed')['qualified'])
            frames[15]['ai_quality']=None
            (root/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
            with self.assertRaises(ValueError):qualify.native_metrics(root,scene,'fixed')

    def test_raw_case_rejects_fallback_and_relaxed_stopping(self):
        scene={'steps':16,'warmup_steps':15}
        policy={'schema':'rheon-learned-stopping-policy-v2','relative_residual':1e-6,
                'absolute_residual':1e-12,'predicted_divergence':1e-7,'mean_free_pressure_relative_error':1e-6}
        receipt={'scene':scene,'binary_sha256':'fixed','status':'completed','error':None,
                 'result':{'hard_stop':False,'ai_steps':1,'qualified_ai_steps':1},
                 'native_process':{'reaped':True,'returncode':0},'cpu_threads':1,'cuda_initialized':False,
                 'native_pressure_refinement':False,'native_fallback':False,'adaptive_stopping':policy,
                 'proposal_network_evaluations':1,'learned_proposals_per_pressure_solve':32,
                 'minimum_network_evaluations':0}
        quality={'qualified':True,'hard_stop':False,'actual_divergence':0.,'courant':0.}
        frames=[{'step':i,'reference_quality':quality,'ai_quality':quality if i==16 else None,
                 'relative_pressure_difference':1e-7} for i in range(1,17)]
        last={'proposals':1,'equation_passed':True,'pressure_error_bound_passed':True,
              'residual_l2':1e-12,'rhs_l2':1.,'predicted_divergence':1e-9,
              'mean_free_pressure_relative_error_bound':1e-7}
        callback={'step':16,'network_passes':1,'proposals':1,'trace':[last]}
        diagnostic={'samples':[{'step':16,'oracle_independent_relative_defect':1e-14,
                                'pressure_bound_covers_oracle':True}]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'native').mkdir()
            def save():
                (root/'native/receipt.json').write_text(json.dumps(receipt))
                (root/'native/frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
                (root/'native/callbacks.jsonl').write_text(json.dumps(callback)+'\n')
                (root/'diagnostics.json').write_text(json.dumps(diagnostic))
            save()
            self.assertTrue(qualify.case_metrics(root,scene,{'scene':0,'group':'original58','expected_ai_steps':1},policy,'fixed')['qualified'])
            receipt['native_fallback']=True;save()
            with self.assertRaises(ValueError):qualify.case_metrics(root,scene,{'expected_ai_steps':1},policy,'fixed')
            receipt['native_fallback']=False;last['predicted_divergence']=1.00000001e-7;save()
            with self.assertRaises(ValueError):qualify.case_metrics(root,scene,{'expected_ai_steps':1},policy,'fixed')

    def test_canonical_hash_binds_order_and_terminal_newline(self):
        self.assertEqual(qualify.canonical([2,1]),b'[2,1]\n')
        self.assertNotEqual(qualify.sha(qualify.canonical([2,1])),qualify.sha(qualify.canonical([1,2])))

    def test_cohort_admits_actual_current_runtime_producer_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            # Exercise the real receipt producer with a tiny stdlib-only owner.
            (root/'torch_runtime.py').write_text(
                'import json\n'
                'def inventory(roots): return {**roots,"files":[],"file_links":[],"total_bytes":0,"assumption":"test fixture"}\n'
                'def verify(value): assert value["files"] == []\n'
                'def encoded(value): return json.dumps(value).encode()\n')
            baseline=root/'baseline.json'
            baseline.write_text(json.dumps({'python':'fixed-python','python_root':'fixed-root',
                                           'site_packages':'fixed-packages','files':[],'total_bytes':0}))
            class Selection:
                def source(self,name):
                    assert name=='tuldok'
                    return root
            receipt=dependency_runtime.measure(Selection(),baseline,qualify.digest(baseline),root/'measured')
            self.assertIsNone(qualify_cohort.admit_runtime_receipt(receipt))
            self.assertNotIn('all_baseline_fields_and_asset_bytes_identical',receipt)
            historical={**receipt,'all_baseline_fields_and_asset_bytes_identical':True}
            del historical['all_original_fields_and_asset_bytes_identical']
            with self.assertRaises(KeyError):qualify_cohort.admit_runtime_receipt(historical)
            with self.assertRaises(ValueError):
                qualify_cohort.admit_runtime_receipt({**receipt,'full_second_verification':False})


if __name__=='__main__':unittest.main()
