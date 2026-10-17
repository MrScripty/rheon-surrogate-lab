"""Negative fixtures for the admitted source and new-tensor boundary."""
import copy
import unittest
from compare import architecture_contract


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.witness = {'sources':{'model.py':'model-hash','coarse.py':'coarse-hash'}}
        self.qualification = {'status':'passed','device':'cpu','sources':{
            '/qualified/overlay/model.py':'model-hash','/qualified/overlay/coarse.py':'coarse-hash'}}
        self.tensor = {'shape':[64,64],'dtype':'<f4','bytes':16384,'nonzero':0}

    def test_admitted_architecture(self):
        architecture_contract(self.witness,self.qualification,self.tensor)

    def test_different_new_source_is_rejected(self):
        value = copy.deepcopy(self.witness)
        value['sources']['coarse.py'] = 'unqualified-source'
        with self.assertRaisesRegex(ValueError,'candidate source differs from qualified overlay: coarse.py'):
            architecture_contract(value,self.qualification,self.tensor)

    def test_wrong_new_shape_is_rejected(self):
        value = {**self.tensor,'shape':[32,128]}
        with self.assertRaisesRegex(ValueError,'exact zero float32 64-by-64 coarse tensor required'):
            architecture_contract(self.witness,self.qualification,value)

    def test_nonzero_and_wrong_precision_are_rejected(self):
        for update in ({'nonzero':1},{'dtype':'<f8','bytes':32768}):
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'exact zero float32 64-by-64 coarse tensor required'):
                architecture_contract(self.witness,self.qualification,{**self.tensor,**update})


if __name__ == '__main__':
    unittest.main()
