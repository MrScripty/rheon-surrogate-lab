"""Hand-computed tradeoffs exercise selection independently of the dashboard."""
import unittest
from material import selection


class SelectionTests(unittest.TestCase):
    def model(self,identifier,time,error,**overrides):
        result=dict(id=identifier,order=int(identifier[1:]),weights_sha256=identifier,cases=128,
            worst_error=error,inference_median_ms=time,timing_cases=[dict(trajectory_id='v'+str(i),native_frame=15,
                counts_xyz=[8,8,8],cell_size_m=.125,split='validation')for i in range(128)])
        return {**result,**overrides}

    def test_union_keeps_frontier_and_ranked_groups_without_dominated_model(self):
        models=[self.model('C01',1,4),self.model('C02',2,2),self.model('C03',3,3),self.model('C04',5,1)]
        result={c['id']:tags for c,tags in selection(models,2)}
        self.assertEqual(result,{'C01':['frontier','fast'],'C02':['frontier','accurate','fast'],'C04':['frontier','accurate']})

    def test_other_populations_do_not_enter_rankings(self):
        valid=self.model('C01',2,2)
        other=self.model('C02',1,1);other['timing_cases'][0]['native_frame']=63
        not_validation=self.model('C03',.1,.1);not_validation['timing_cases'][0]['split']='train'
        self.assertEqual([c['id']for c,tags in selection([valid,other,not_validation],10)],['C01'])

    def test_repeated_checkpoint_keeps_one_entry_and_combines_memberships(self):
        first=self.model('C01',1,4)
        duplicate=self.model('C02',4,1,weights_sha256='C01')
        result=selection([first,duplicate,self.model('C03',2,2)],1)
        self.assertEqual([(c['id'],tags)for c,tags in result],
                         [('C01',['frontier','accurate','fast']),('C03',['frontier'])])


if __name__=='__main__':unittest.main()
