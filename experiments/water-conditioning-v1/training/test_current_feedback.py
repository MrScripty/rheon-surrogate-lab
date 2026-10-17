"""Current-policy feedback changes supervision without changing inference."""
import copy
import unittest
import torch
import analytical
import data
import direction_feedback
import model
import policy
import test_residual
import train


class CurrentFeedback(unittest.TestCase):
    def setUp(self):
        fixture=test_residual.Residual();fixture.setUp();self.sample=fixture.sample;self.norm=fixture.norm
        self.old=policy.decode(policy.read(policy.LAB/'experiments/pressure-throughput-feedback-v1/configs/weight003.json'))
        self.config={**copy.deepcopy(self.old),'schema':'rheon-water-residual-experiment-v11',
            'current_feedback':{'stage':'late','weight':.1,'seed':73,'manifest_sha256':'a'*64}}

    def test_selected_input_inventory_and_exact_inference_initialization(self):
        policy.experiment(self.config)
        self.assertEqual(policy.execution(self.config),policy.execution(self.old))
        self.assertEqual(set(train.input_slots(self.config)),set(data.DATA)|set(direction_feedback.SLOTS)|{'current_dataset','current_manifest'})
        for field,value in [('stage','unknown'),('weight',float('nan')),('seed',True)]:
            invalid=copy.deepcopy(self.config);invalid['current_feedback'][field]=value
            with self.assertRaisesRegex(ValueError,'direction_feedback'):policy.experiment(invalid)
        model.select(self.old);torch.manual_seed(47);before=model.PressureCorrectionNet()
        model.select(self.config);torch.manual_seed(47);after=model.PressureCorrectionNet()
        for a,b in zip(before.state_dict().values(),after.state_dict().values()):self.assertTrue(torch.equal(a,b))
        torch.testing.assert_close(model.predict_pressure(before,self.sample,self.norm),model.predict_pressure(after,self.sample,self.norm),rtol=0,atol=0)

    def test_older_project_builder_rejects_current_bank_before_reading_packets(self):
        import project
        from unittest.mock import patch
        path=policy.LAB/'experiments/pressure-current-policy-v1/configs/early.json'
        with patch.object(project.subprocess,'check_output',side_effect=[b'',path.read_bytes()]),patch.object(project.data,'read_view',side_effect=AssertionError('must reject before reading packet')):
            with self.assertRaisesRegex(ValueError,'current_policy_project'):
                project.assemble(None,None,None,None,None,None,path)

    def test_native_amplitude_new_gradient_adds_to_existing_direction_stream(self):
        model.select(self.config);torch.manual_seed(73);expected=model.PressureCorrectionNet();actual=copy.deepcopy(expected)
        old=analytical.make_sample(self.sample,[((0,0,1),1.)],2.)
        late=analytical.make_sample(self.sample,[((0,1,0),1.)],2e-8)
        model.training_loss(expected,self.sample,self.norm).mean().backward()
        for sample in (old,late):
            (.1*model.training_loss(expected,sample,self.norm,
                criterion=lambda p:analytical.loss(p,sample,model.full_operator)).mean()).backward()
        wanted=torch.nn.utils.clip_grad_norm_(expected.parameters(),1.,error_if_nonfinite=True)
        _,observed=train.one_update(actual,torch.optim.SGD(actual.parameters(),lr=0.),[],[(0,0)],self.norm,
            cache={(0,0):self.sample},direction_sample=old,direction_weight=.1,current_sample=late,current_weight=.1)
        self.assertEqual(float(wanted),observed)
        for a,b in zip(expected.parameters(),actual.parameters()):self.assertTrue(torch.equal(a.grad,b.grad))

if __name__=='__main__':unittest.main()
