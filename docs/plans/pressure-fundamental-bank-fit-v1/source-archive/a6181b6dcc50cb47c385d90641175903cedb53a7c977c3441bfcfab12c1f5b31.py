"""Verify completed study evidence, retained source identities and numerical gates."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def verify(directory,output):
    result=read(directory/'results.json')
    assert result['status']=='completed'
    checks=[]
    for name,identity in result['sources'].items():assert sha(ROOT/name)==identity
    for name,arm in result['arms'].items():
        training=arm['training'][0];identifier=training['id'];bank=training['fundamental_bank'];count=len(bank['cases'])
        for source,identity in arm['sources'].items():assert sha(ROOT/source)==identity
        assert bank['expected_exposures']==bank['optimizer_exposures']==[200]*count
        assert training['updates']==200*count
        assert training['optimizer_work']['network_forwards']==800*count
        assert training['optimizer_work']['forward_cells']==800*sum(math.prod(c['counts_xyz'])for c in bank['cases'])
        assert len(bank['trace'])==20
        assert all(len(t['measurements'])==4*count for t in bank['trace'])
        assert all(math.isfinite(r['relative_equation_residual'])for t in bank['trace']for r in t['measurements'])
        assert all(training['artifact_parity'].values())
        assert arm['cpu_execution']['all_owned_children_reaped']
        for model in ['C53',identifier]:
            fitted=arm['fitted'][model];snapshots=arm['snapshots'][model]
            assert len(fitted['measurements'])==4*count and len(snapshots['rows'])==512
            assert len(arm['diagnostics'][model]['rows'])==960
            assert fitted['threads']==snapshots['torch_threads']==1
            assert not fitted['cuda_initialized'] and not snapshots['cuda_initialized']
            assert all(math.isfinite(r['relative_equation_residual'])for r in fitted['measurements'])
            assert all(math.isfinite(r['relative_residual'])for r in snapshots['rows'])
        keys=lambda m:{(r['case'],r['passes']):r['relative_equation_residual']for r in arm['fitted'][m]['measurements']}
        b=keys('C53');c=keys(identifier)
        gate=all(c[i,n]<=1e-6 for i in range(count)for n in [4,8])and all(c[i,n]<=b[i,n]for i in range(count)for n in [1,2])
        assert gate==arm['decision']['capacity_gate_passed']
        assert arm['historical_baseline_quality_exact_rows']==512
        assert not arm['decision']['model_promoted']
        checks.append(dict(arm=name,model=identifier,cases=count,optimizer_updates=training['updates'],
            capacity_gate_passed=gate,restored_cpu_fitted_rows=4*count,real_rows=512,manufactured_rows=960,
            exact_initial_inventory=True,artifact_parity=True,source_identity=True,all_owned_cpu_workers_reaped=True))
    capacity=result['arms']['combined-fundamentals']['decision']['capacity_gate_passed']
    assert len(result['arms'])==(1 if capacity else 3)
    assert result['decision']['baseline_retained']=='C53' and not result['decision']['model_promoted']
    write(output,dict(status='passed',scope='Execution and evidence verification; scientific capacity gates recorded separately.',
        checks=checks,original_untouched_holdouts_consumed=0,source_sha256=sha(__file__),results_sha256=sha(directory/'results.json')))
    print('Verified',len(checks),'completed screens',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.directory.resolve(),args.output.resolve())
