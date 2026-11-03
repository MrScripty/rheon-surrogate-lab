"""Provenance cases where equal weights must not manufacture a fit or lineage."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('worklist', Path(__file__).with_name('model_worklist.py'))
worklist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worklist)


def fit(name, path, weight='a' * 64, run_id=None):
    return dict(item_id=name, receipt_paths=[path], named_entries=[],
                recorded_Run_observations=[dict(run_id=run_id)] if run_id else [],
                checkpoints=[dict(sha256=weight, stage='selected')])


def declaration(record, weight='a' * 64):
    return dict(discovery_key='cohorts/0', source_pointer='/cohort_points/0', record=record,
                metadata=dict(weights_sha256=weight), checkpoint_disposition='available_content_verified',
                checkpoint_path='runs/retained/weights.npz')


class ModelWorklistTests(unittest.TestCase):
    def test_specific_replay_beats_broad_shared_run_root(self):
        fits = [fit('wanted', 'runs/study/arm-A/train/receipt.json'),
                fit('other-fit-same-weights', 'runs/study/arm-B/train/receipt.json')]
        record = dict(id='C78', replay_artifacts='../../runs/study/arm-A/restore', run_root='../../runs/study')
        result = worklist.link_entries(dict(declarations=[declaration(record)]), fits)[0]
        self.assertEqual(result['fit_item_id'], 'wanted')

    def test_equal_weights_without_context_remain_ambiguous(self):
        fits = [fit('one', 'runs/one/train/receipt.json'), fit('two', 'runs/two/train/receipt.json')]
        result = worklist.link_entries(dict(declarations=[declaration(dict(id='unknown'))]), fits)[0]
        self.assertIsNone(result['fit_item_id'])
        self.assertEqual(result['candidate_fit_item_ids'], ['one', 'two'])

    def test_composition_links_actual_parents_without_inventing_new_fit(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(worklist, 'ROOT', Path(folder)):
            path = Path(folder) / 'runs/assembly/receipt.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='assembled_from_exact_managed_CPU_parents',
                weights_sha256='c' * 64, parents={
                    'parent': dict(training_run_id='r1', weights_sha256='a' * 64),
                    'component': dict(training_run_id='r2', weights_sha256='b' * 64)})))
            fits = [fit('parent-fit', 'runs/parent/train/receipt.json', 'a' * 64, 'r1'),
                    fit('component-fit', 'runs/component/train/receipt.json', 'b' * 64, 'r2')]
            record = dict(id='composition', record_kind='composition', optimizer_updates=0,
                          weights_sha256='c' * 64, source='../../runs/assembly/receipt.json')
            result = worklist.link_entries(dict(declarations=[declaration(record, 'c' * 64)]), fits)[0]
            self.assertEqual(result['fit_link_disposition'], 'recorded_composition_of_exact_parent_checkpoints')
            self.assertIsNone(result['fit_item_id'])
            self.assertEqual([p['fit_item_ids'] for p in result['checkpoint_assembly']['parents']],
                             [['parent-fit'], ['component-fit']])


if __name__ == '__main__':
    unittest.main()
