"""Captured paired-bank identity, private sampling and additive gradient proof."""
import copy
import io
import unittest
import numpy as np
import torch
import analytical
import data
import direction_feedback
import model
import policy
import test_residual
import train


class Directions(unittest.TestCase):
    def setUp(self):
        self.fixture=test_residual.Residual();self.fixture.setUp()
        self.config=policy.decode(policy.read(policy.LAB/'experiments/water-targeted-feedback-v1/configs/targeted-feedback.json'))
        self.spec={'stage':'early','weight':.1,'seed':71,'manifest_sha256':'a'*64}

    def payload(self):
        samples=[analytical.make_sample(self.fixture.sample,[((0,0,1),1.)],2.),
                 analytical.make_sample(self.fixture.sample,[((0,1,0),1.)],2.)]
        arrays={f'case_{i:03d}_{k}':s[k].numpy().copy() for i,s in enumerate(samples) for k in ('rhs','reference')}
        stream=io.BytesIO();np.savez(stream,**arrays);numeric=stream.getvalue()
        cases=[{'identity':{'kind':'real','group':f'group-{i}','trajectory_id':'train-0'},
            'counts_xyz':list(s['base'].shape[2:][::-1]),'rho':1000.,'dt':1/60,'cell_size_m':.125,
            'stages':{'early':{'operator_relative_error':0.},'late':{'operator_relative_error':0.}}} for i,s in enumerate(samples)]
        manifest={'schema':'rheon-pressure-direction-bank-v1','source_corrections':8,'optimizer_updates':0,
            'train_trajectory_ids':['train-0'],'packet_splits_sha256':policy.sha(b'{}'),
            'source':{'normalization_sha256':policy.sha(policy.canonical(self.fixture.norm))},'amplitude_rms_pa':2.,
            'cases':cases,'stages':{k:{'bytes':len(numeric),'sha256':policy.sha(numeric),'arrays':data.inventory(arrays)} for k in ('early','late')}}
        raw=policy.canonical(manifest)
        return {'direction_manifest':raw,'direction_dataset':numeric,'splits':b'{}'}, {**self.spec,'manifest_sha256':policy.sha(raw)}

    def test_closed_version_and_legacy_input_inventory(self):
        old=copy.deepcopy(self.config)
        self.assertEqual(set(train.input_slots(old)),set(data.DATA))
        with self.assertRaisesRegex(ValueError,'residual_configuration'):
            policy.experiment({**old,'direction_feedback':self.spec})
        config={**old,'schema':'rheon-water-residual-experiment-v8','direction_feedback':self.spec}
        policy.experiment(config)
        self.assertEqual(set(train.input_slots(config)),set(data.DATA)|set(direction_feedback.SLOTS))
        for change in ({'stage':'unknown'},{'seed':True},{'weight':float('nan')},{'manifest_sha256':'invalid'},{'unknown':1}):
            invalid=copy.deepcopy(config);invalid['direction_feedback'].update(change)
            with self.assertRaisesRegex(ValueError,'direction_feedback'):policy.experiment(invalid)

    def test_stage_identity_sampling_and_combined_memory(self):
        inputs,spec=self.payload()
        state=torch.random.get_rng_state().clone()
        np.random.seed(32);expected=np.random.random(2);np.random.seed(32)
        first,order,evidence=direction_feedback.bank(inputs,spec,self.fixture.norm,['train-0'],31,'cpu',123,1024**2)
        second,again,other=direction_feedback.bank(inputs,{**spec,'stage':'late'},self.fixture.norm,['train-0'],31,'cpu',123,1024**2)
        self.assertTrue(np.array_equal(order,again));self.assertEqual(sorted(evidence['group_exposures']),[15,16])
        self.assertEqual(evidence['forward_cells_per_correction'],other['forward_cells_per_correction'])
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()));self.assertTrue(np.array_equal(expected,np.random.random(2)))
        for a,b in zip(first,second):
            for key in a:self.assertTrue(torch.equal(a[key],b[key]))
        with self.assertRaisesRegex(ValueError,'direction_cache'):
            direction_feedback.bank(inputs,spec,self.fixture.norm,['train-0'],31,'cpu',123,evidence['combined_resident_tensor_bytes']-1)
        with self.assertRaisesRegex(ValueError,'direction_payload'):
            direction_feedback.bank({**inputs,'direction_dataset':inputs['direction_dataset']+b'x'},spec,self.fixture.norm,['train-0'],31,'cpu',0,1024**2)
        with self.assertRaisesRegex(ValueError,'direction_lineage'):
            direction_feedback.bank(inputs,spec,self.fixture.norm,['held-out'],31,'cpu',0,1024**2)

    def test_additive_gradient_and_legacy_prediction_parity(self):
        old=copy.deepcopy(self.config);model.select(old);torch.manual_seed(47)
        before=model.PressureCorrectionNet()
        config={**old,'schema':'rheon-water-residual-experiment-v8','direction_feedback':self.spec}
        model.select(config);torch.manual_seed(47);after=model.PressureCorrectionNet()
        for a,b in zip(before.state_dict().values(),after.state_dict().values()):self.assertTrue(torch.equal(a,b))
        torch.testing.assert_close(model.predict_pressure(before,self.fixture.sample,self.fixture.norm),
            model.predict_pressure(after,self.fixture.sample,self.fixture.norm),rtol=0,atol=0)
        first=copy.deepcopy(after);second=copy.deepcopy(after)
        extra=analytical.make_sample(self.fixture.sample,[((0,0,1),1.)],2.)
        model.training_loss(first,self.fixture.sample,self.fixture.norm).mean().backward()
        (.1*model.training_loss(first,extra,self.fixture.norm,
            criterion=lambda p:analytical.loss(p,extra,model.full_operator)).mean()).backward()
        norm=torch.nn.utils.clip_grad_norm_(first.parameters(),1.,error_if_nonfinite=True)
        _,actual=train.one_update(second,torch.optim.SGD(second.parameters(),lr=0.),[],[(0,0)],self.fixture.norm,
            cache={(0,0):self.fixture.sample},direction_sample=extra,direction_weight=.1)
        self.assertEqual(float(norm),actual)
        for a,b in zip(first.parameters(),second.parameters()):self.assertTrue(torch.equal(a.grad,b.grad))


if __name__=='__main__':unittest.main()
