"""Check receipt identity, measured work and the predeclared conditional gate."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def verify(directory,output):
    result=read(directory/'results.json');assert result['status']=='completed'
    for name,identity in result['sources'].items():assert sha(ROOT/name)==identity
    checks=[]
    for identifier,g in result['gradients'].items():
        assert g['status']=='passed' and g['inventory_unchanged'] and g['optimizer_updates']==0
        assert g['objective_checkpoint']=='C66' and g['threads']==1 and not g['cuda_initialized']
        assert g['gradient_vector_bytes']<=16*1024**2 and g['teacher_forwards']==0
        assert len(g['cases'])==8 and len(g['cosine_matrix'])==8
        for i in range(8):
            assert g['cases'][i]['gradient_l2']>0
            for j in range(8):
                cosine=g['cosine_matrix'][i][j];dot=g['dot_matrix'][i][j]
                assert math.isfinite(cosine)and -1.000000000001<=cosine<=1.000000000001
                assert cosine==g['cosine_matrix'][j][i]
                assert math.isclose(cosine,dot/g['cases'][i]['gradient_l2']/g['cases'][j]['gradient_l2'],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(g['cosine_matrix'][i][i],1,abs_tol=1e-12)
        assert len(g['negative_pairs'])==sum(g['cosine_matrix'][i][j]<0 for i in range(8)for j in range(i+1,8))
        for name,identity in g['sources'].items():assert sha(ROOT/name)==identity
        checks.append(dict(model=identifier,gradient_cases=8,opposing_pairs=len(g['negative_pairs']),inventory_unchanged=True))
    for name,projection in result['arms'].items():
        path=directory/'reports'/name/'result.json';arm=read(path);t=arm['training'][0];identifier=t['id'];bank=t['fundamental_bank'];count=len(bank['cases'])
        for name,identity in {**arm['sources'],**arm['evidence_sources']}.items():assert sha(ROOT/name)==identity
        assert bank['expected_exposures']==bank['optimizer_exposures']==[200]*count
        assert t['updates']==200*count and t['optimizer_work']['network_forwards']==800*count
        assert t['optimizer_work']['forward_cells']==800*sum(math.prod(c['counts_xyz'])for c in bank['cases'])
        assert all(t['artifact_parity'].values())and all(r['state']=='completed'and r['custody_phase']=='drained'for r in t['managed_custody'])
        assert arm['cpu_execution']['status']=='completed'and arm['cpu_execution']['all_owned_children_reaped']
        for model in ['C53',identifier]:
            f,s,d=arm['fitted'][model],arm['snapshots'][model],arm['diagnostics'][model]
            assert len(f['measurements'])==4*count and len(s['rows'])==512 and len(d['rows'])==960
            assert not any(r['cuda_initialized']for r in [f,s,d])
            assert f['threads']==s['torch_threads']==1
            assert all(math.isfinite(r['relative_residual'])for r in s['rows'])
            assert all(math.isfinite(r['relative_equation_residual'])for r in f['measurements']+d['rows'])
        values=lambda model:{(r['case'],r['passes']):r['relative_equation_residual']for r in arm['fitted'][model]['measurements']}
        b,c=values('C53'),values(identifier)
        gate=all(c[i,n]<=1e-6 for i in range(count)for n in [4,8])and all(c[i,n]<=b[i,n]for i in range(count)for n in [1,2])
        assert gate==arm['decision']['capacity_gate_passed']
        match=arm['matched_comparison'];prior=read(ROOT/match['control_result_source'])
        assert sha(ROOT/match['control_result_source'])==match['control_result_sha256']
        control=prior['training'][0];assert t['optimizer_work']==control['optimizer_work']
        assert bank['sampling_order_sha256']==control['fundamental_bank']['sampling_order_sha256']
        prefix=lambda trace:[r for r in trace if r['update']<=100*count]
        assert prefix(bank['trace'])==prefix(control['fundamental_bank']['trace'])
        assert match['exact_shared_cuda_prefix_updates']==100*count
        assert arm['historical_baseline_quality_exact_rows']==512 and not arm['decision']['model_promoted']
        assert projection['training']==t and projection['decision']==arm['decision']
        checks.append(dict(model=identifier,cases=count,updates=t['updates'],fitted_gate=gate,real_rows=512,manufactured_rows=960,exact_cuda_prefix=100*count))
    assert len(result['arms'])==(2 if result['arms']['shape-later-decay']['decision']['capacity_gate_passed']else 1)
    assert result['decision']['baseline_retained']=='C53'and not result['decision']['model_promoted']
    write(output,dict(status='passed',scope='Evidence consistency and execution verification; scientific fitting gate recorded separately.',checks=checks,
        results_sha256=sha(directory/'results.json'),source_sha256=sha(__file__),untouched_holdouts_consumed=0))
    print('Verified completed matched schedule study',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.directory.resolve(),a.output.resolve())
