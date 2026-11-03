"""Selected-state and source-identity boundaries without any neural runtime."""
import copy
from pathlib import Path
import tempfile
import unittest

from component_material import selected_inventory,observed_file


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {'weight':dict(dtype='<f4',shape=[3,3,3,3,3],sha256='a'*64)}

    def test_selected_final_receipt_and_history_must_agree(self):
        actual,_ = selected_inventory({'final_inventory':self.inventory},{'final_inventory':self.inventory},'b'*64)
        self.assertEqual(actual,self.inventory)
        changed = copy.deepcopy(self.inventory);changed['weight']['sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError,'inventories differ'):
            selected_inventory({'final_inventory':self.inventory},{'final_inventory':changed},'b'*64)

    def test_guided_selected_not_best_mean_or_initial(self):
        changed = copy.deepcopy(self.inventory);changed['weight']['sha256'] = 'c'*64
        value = {'checkpoint_inventories':{'weights.npz':self.inventory,'best-mean-weights.npz':changed}}
        actual,_ = selected_inventory(value,value,'b'*64)
        self.assertEqual(actual,self.inventory)

    def test_same_archive_aliases_allowed_only_with_equal_inventories(self):
        history = {'checkpoints':{'a':dict(sha256='b'*64,inventory=self.inventory),'b':dict(sha256='b'*64,inventory=self.inventory)}}
        self.assertEqual(selected_inventory({},history,'b'*64)[0],self.inventory)
        history['checkpoints']['b']['inventory'] = {'other':dict(dtype='<f8',shape=[1],sha256='c'*64)}
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            selected_inventory({},history,'b'*64)

    def test_changed_bytes_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'material';path.write_bytes(b'original')
            binding = observed_file(root,'material')
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'bytes differ'):
                observed_file(root,'material',binding)
            with self.assertRaisesRegex(ValueError,'inside'):
                observed_file(root,'../material')


if __name__ == '__main__':
    unittest.main()
