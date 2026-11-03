"""Publication gating and exact retained-archive identity, without Torch."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import import_common_generator_measurements as common
import original_train_pressure_accuracy as worker


class PublicationGate(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.ui_path,self.audit_path,self.header_path = root/'ui.json',root/'custody.json',root/'header.json'
        self.checkpoint = 'a'*64
        self.header_path.write_text(json.dumps(dict(checkpoint=self.checkpoint)))
        self.ui = dict(status='matched39configurations_and35outcome_notes_actual_native_UI_accepted',
            browser_stopped=True,captures_actually_viewed=[f'image-{i}.png' for i in range(10)],
            requested_chart_comparisons=40,displayed_numeric_presence_assertions=390,checkpoint=self.checkpoint)
        self.audit = dict(status='native38matched_outcome_notes_exact_and_scientific_records_unchanged',
            checkpoint=self.checkpoint,canonical_header_sha256=hashlib.sha256(self.header_path.read_bytes()).hexdigest())

    def invoke(self):
        self.ui_path.write_text(json.dumps(self.ui));self.audit_path.write_text(json.dumps(self.audit))
        return worker.publication_gate(self.ui_path,self.audit_path,self.header_path)

    def test_matching_accepted_report_and_header_identity_pass_metadata_gate(self):
        self.assertEqual(self.invoke()['checkpoint'],self.checkpoint)

    def test_running_browser_or_uninspected_captures_reject(self):
        self.ui['browser_stopped'] = False
        with self.assertRaisesRegex(ValueError,'^Complete real chart/summary/image acceptance required$'):
            self.invoke()
        self.ui['browser_stopped'] = True
        self.ui['captures_actually_viewed'] = []
        with self.assertRaisesRegex(ValueError,'^Complete real chart/summary/image acceptance required$'):
            self.invoke()

    def test_changed_header_rejects_even_with_same_checkpoint_text(self):
        self.header_path.write_text(json.dumps(dict(checkpoint=self.checkpoint,unaccepted_change=True)))
        with self.assertRaisesRegex(ValueError,'^Current native project differs from accepted publication checkpoint$'):
            self.invoke()

    def test_unavailable_publication_stops_before_protocol_or_model_import(self):
        with patch.object(worker,'publication_gate',side_effect=ValueError('publication unavailable')), \
                patch.object(worker,'read',side_effect=AssertionError('Protocol must not be read yet')):
            with self.assertRaisesRegex(ValueError,'^publication unavailable$'):
                worker.run(Path('not-opened.json'),'fit-not-loaded')


class RetainedArchive(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root/'pressure.npz'
        self.archive.write_bytes(b'opaque original archive identity fixture')
        self.result_path = self.root/'result.json'
        self.rows = [dict(trajectory_id='fixture-scene',parent=i//2,native_frame=15 if i%2==0 else 63,
            counts_xyz=[2,2,2],cell_size_m=.125,input_sha256={'base':'input'},
            reference_pressure_sha256='pressure-reference',reference_rhs_sha256='rhs-reference',output_sha256=f'output-{i}') for i in range(128)]
        self.prior = dict(status='completed_captured_pressure_GPU_accuracy',weights_sha256='same-weights',rows=self.rows,
            neural_module_forward_calls={'cpu':0,'cuda':128},learned_parameters_unchanged=True,architecture_unchanged=True,
            output_archive=dict(path='pressure.npz',sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest()))
        self.result_path.write_text(json.dumps(self.prior))
        self.protocol = dict(prior_controls={'fit-test':[dict(identifier='original-control',result=self.spec(self.result_path),
            archive=self.spec(self.archive))]})

    def spec(self,path):
        return dict(path=str(path.relative_to(self.root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def invoke(self,rows=None):
        with patch.object(worker,'ROOT',self.root),patch.object(common,'ROOT',self.root):
            return worker.matching_archive(self.protocol,'fit-test',self.rows if rows is None else rows,'same-weights')

    def test_all128_exact_case_hashes_reuse_only_original_archive(self):
        before = set(self.root.iterdir())
        archive,comparisons = self.invoke()
        self.assertEqual(archive['kind'],'existing_original_prediction_archive')
        self.assertEqual(archive['path'],'pressure.npz')
        self.assertTrue(comparisons[0]['all128_request_reference_and_output_hashes_exact'])
        self.assertFalse(comparisons[0]['general_policy_equivalence_qualified'])
        self.assertEqual(set(self.root.iterdir()),before)

    def test_one_changed_reference_or_output_prevents_reuse(self):
        for field in ['reference_rhs_sha256','output_sha256']:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[-1][field] = 'changed'
                archive,comparisons = self.invoke(rows)
                self.assertIsNone(archive)
                self.assertFalse(comparisons[0]['all128_request_reference_and_output_hashes_exact'])

    def test_another_validly_hashed_archive_cannot_replace_the_original(self):
        unrelated = self.root/'unrelated.npz'
        unrelated.write_bytes(b'different scientific payload')
        self.protocol['prior_controls']['fit-test'][0]['archive'] = self.spec(unrelated)
        with self.assertRaisesRegex(ValueError,'^Archive must belong to the matched original result$'):
            self.invoke()


if __name__ == '__main__':
    unittest.main()
