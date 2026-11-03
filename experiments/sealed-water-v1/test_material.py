import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from policy import HERE,LAB,decode,load_module,read
import data
import model
import train


def parents():
    result=[]
    for number,split in enumerate(('train','validation','reserved_test')):
        fields={key:np.full((2,3,4,5),.1+number,dtype='<f8') for key in data.FIELDS}
        fields['base'][:,0,0,0]=0;fields['reference'][:,0,0,0]=0
        result.append({'entry':{'trajectory_id':split,'split':split,'dt_s':1/60},'cell_size_m':.0625,
            'density_kg_m3':1000.0,'counts_xyz':[5,4,3],'frames':[{'index':15},{'index':63}],'arrays':fields})
    return result


class WaterMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):model.configure_cpu()

    def test_real_water_conditioning_and_same_qualified_parameter_initialization(self):
        sample=data.sample(parents()[0],0);self.assertEqual(float(sample['rho'][0]),1000)
        norm=data.fit_normalization(parents());values=model.build_inputs(sample['native_input_residual'],sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],norm)
        self.assertEqual(values.shape,(1,7,3,4,5));self.assertTrue(bool((values[:,5]==1000).all()))
        self.assertEqual(set(norm['scales']),{'residual','pressure','rhs','rho','dt'})
        torch.manual_seed(813);first=model.PressureCorrectionNet()
        torch.manual_seed(813);second=model.owner.PressureCorrectionNet()
        self.assertEqual(model.parameter_count(first),371729)
        self.assertTrue(all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items()))
        pressure=torch.tensor([[[[[0.,2.,5.]]]]],dtype=torch.float64)
        action=model.full_operator(pressure,torch.tensor([.25],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))
        torch.testing.assert_close(action.flatten(),torch.tensor([-.0005,-.00025,.00075],dtype=torch.float64),rtol=0,atol=1e-18)

    def test_reserved_and_validation_values_never_affect_scales_or_sample_order(self):
        rows=parents();normalization=data.fit_normalization(rows)
        for parent in rows[1:]:
            for value in parent['arrays'].values():value.fill(1e15)
        self.assertEqual(normalization,data.fit_normalization(rows));self.assertEqual(normalization['training_provenance']['frames'],2)
        config=decode(read(HERE/'configs/water.json'))['training'];config['training']['updates']=100
        order=train.order(rows,config);self.assertTrue(set(order[:,:,0].flatten())=={0})
        self.assertTrue(set(order[:,:,1].flatten())=={0,1})
        rows[0]['frames'].append({'index':31})
        self.assertTrue(set(train.order(rows,config)[:,:,1].flatten())=={0,1,2})

    def test_witnesses_are_bounded_and_exclude_reserved_test(self):
        rows=parents();cases=train.cases(rows)
        self.assertEqual(len(cases),2);self.assertTrue(all(c['split']!='reserved_test' for c in cases))
        for case in cases:self.assertEqual(case['native_frame'],rows[case['parent']]['frames'][case['frame']]['index'])


if __name__=='__main__':unittest.main()
