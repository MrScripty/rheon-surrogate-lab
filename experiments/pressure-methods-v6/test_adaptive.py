"""Economic admission, host commit and shape/forcing resets use no labels."""
import tempfile,unittest
from pathlib import Path
from v6_support import write
from adaptive import AdaptiveRule
import torch
class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2026101914)
        self.temp=tempfile.TemporaryDirectory(prefix='rheon-v6-adaptive-');self.addCleanup(self.temp.cleanup)
        self.r=torch.randn((1,1,5,5,5),dtype=torch.float64);self.key=('scene',tuple(self.r.shape),.0625,1000.)
    def rule(self,overhead=.01):
        p=Path(self.temp.name)/('cost-'+str(overhead)+'.json')
        write(p,dict(status='frozen_from_consumed_development',minimum_cosine=.9,overhead_margin=1.1,
          bands=[dict(cells=125,baseline_iterations=4,saved_fraction=.25,cycle_seconds_per_cell=.1,overhead_seconds_per_cell=overhead)],
          unit_fixture=True))
        return AdaptiveRule(p)
    def accept(self,rule,work=4):rule.observe(dict(coarse_calls=work));rule.commit(True)
    def test_positive_economics_cold_then_similar(self):
        rule=self.rule();cold=rule.decide(self.r,self.key);self.assertTrue(cold['collect_basis']);self.assertFalse(cold['admitted']);self.accept(rule)
        warm=rule.decide(2*self.r,self.key);self.assertTrue(warm['admitted']);self.assertGreater(warm['expected_savings_seconds'],warm['expected_overhead_seconds']);self.accept(rule)
    def test_negative_economics_never_collects_or_projects(self):
        rule=self.rule(overhead=10)
        for _ in range(3):
            d=rule.decide(self.r,self.key);self.assertFalse(d['admitted']);self.assertFalse(d['collect_basis']);self.accept(rule)
    def test_abrupt_reversal_and_changed_context_reset(self):
        rule=self.rule();rule.decide(self.r,self.key);self.accept(rule)
        d=rule.decide(-self.r,self.key);self.assertFalse(d['admitted']);self.accept(rule)
        d=rule.decide(-self.r,('other-scene',*self.key[1:]));self.assertFalse(d['admitted']);self.accept(rule)
    def test_rejected_host_step_has_no_history(self):
        rule=self.rule();rule.decide(self.r,self.key);rule.observe(dict(coarse_calls=4));rule.commit(False);self.assertIsNone(rule.history)
    def test_missing_host_decision_or_observation_is_an_error(self):
        rule=self.rule();rule.decide(self.r,self.key)
        with self.assertRaises(ValueError):rule.decide(self.r,self.key)
        with self.assertRaises(ValueError):rule.commit(True)
        rule.commit(False)
    def test_full_reuse_does_not_erase_fresh_work_estimate(self):
        rule=self.rule();rule.decide(self.r,self.key);self.accept(rule)
        for _ in range(8):
            self.assertTrue(rule.decide(self.r,self.key)['admitted']);self.accept(rule,work=0)
            self.assertEqual(rule.history['baseline_iterations'],4)
    def test_history_is_a_bounded_independent_sketch(self):
        rule=self.rule();rule.decide(self.r,self.key);self.accept(rule);old=rule.history['sketch'].clone();self.r.zero_()
        self.assertTrue(torch.equal(rule.history['sketch'],old));self.assertLessEqual(old.numel(),512)
    def test_zero_and_changed_shape_are_safe(self):
        rule=self.rule();rule.decide(self.r,self.key);self.accept(rule)
        self.assertFalse(rule.decide(torch.zeros_like(self.r),self.key)['admitted']);self.accept(rule)
        changed=torch.ones((1,1,9,7,3),dtype=torch.float64)
        self.assertFalse(rule.decide(changed,('scene',tuple(changed.shape),.0625,1000.))['admitted']);self.accept(rule)
if __name__=='__main__':unittest.main()
