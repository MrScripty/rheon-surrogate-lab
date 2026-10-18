"""Verify identities, geometry properties and explicit diagnostic limitations."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def verify(directory,output):
    r=read(directory/'results.json');assert r['status']=='completed';checks=[];limited=0
    for name,identity in r['sources'].items():assert sha(ROOT/name)==identity
    for identifier,projection in r['models'].items():
        d=read(directory/'reports'/(identifier+'-components.json'))
        assert projection=={k:v for k,v in d.items()if k not in ['vectors','sources']}
        for name,identity in {**d['sources'],**d['vectors']}.items():assert sha(ROOT/name)==identity
        assert len(d['vectors'])==40 and sum((ROOT/name).stat().st_size for name in d['vectors'])<=64*1024**2
        assert d['inventory_unchanged']and d['optimizer_updates']==0 and not d['cuda_initialized']and d['threads']==1
        assert d['teacher_forwards']==0 and d['vector_workspace_bound_bytes']<=16*1024**2 and d['fitting_tensor_bytes']<=8*1024**2
        if identifier in ['C53','C66']:assert d['prior_full_gradient_replay']['exact_full_loss_and_dot_matrix']
        assert d['optimizer_direction']['status']=='unavailable'
        for group in [*d['across'].values(),*(c['geometry']for c in d['cases'])]:
            n=len(group['norms']);assert len(group['dot_matrix'])==len(group['cosine_matrix'])==n
            for i in range(n):
                for j in range(n):
                    dot=group['dot_matrix'][i][j];c=group['cosine_matrix'][i][j]
                    assert math.isfinite(dot)and dot==group['dot_matrix'][j][i]
                    if group['norms'][i]and group['norms'][j]:
                        assert math.isfinite(c)and -1.00000000001<=c<=1.00000000001
                        assert math.isclose(c,dot/group['norms'][i]/group['norms'][j],rel_tol=1e-12,abs_tol=1e-12)
                    else:assert c is None
            pairs={(p['first'],p['second'])for p in group['negative_pairs']}
            assert pairs=={(i,j)for i in range(n)for j in range(i+1,n)if group['cosine_matrix'][i][j]is not None and group['cosine_matrix'][i][j]<0}
        for case in d['cases']:
            assert case['joint_gradient_exact']and case['operator_defect']<1e-10
            assert math.isclose(sum(case['values'].values()),case['full_value'],rel_tol=1e-12,abs_tol=1e-14)
            assert case['separate_gradient_precision_check_passed']==(case['gradient_sum_relative_l2']<=5e-6)
        count=sum(not case['separate_gradient_precision_check_passed']for case in d['cases']);limited+=count
        checks.append(dict(model=identifier,joint_equivalence_cases=8,separate_gradient_precision_limited_cases=count,inventory_unchanged=True,optimizer_state='unavailable'))
    assert limited==r['decision']['separate_gradient_precision_limited_cases']
    assert not r['decision']['training_started']and not r['decision']['model_promoted']and r['decision']['baseline_retained']=='C53'
    assert r['execution']['status']=='completed'and all(w['reaped']and w['returncode']==0 for w in r['execution']['workers'])
    for name in ['math-check.json','joint-guard.json']:
        proof=read(directory/'reports'/name);assert proof['status']=='passed'
        assert proof['measure_source_sha256']==sha(Path(__file__).with_name('measure.py'))
    write(output,dict(status='passed_with_recorded_precision_limits',scope='Evidence identity and consistency; geometry and chain rule have a literal oracle. Not physical solver accuracy or causal/Adam verification.',checks=checks,
        results_sha256=sha(directory/'results.json'),source_sha256=sha(__file__),all_cpu_workers_reaped=True))
    print('Verified 24 joint-gradient measurements; retained',limited,'separate-gradient precision limitations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.directory.resolve(),a.output.resolve())
