"""Independently recompute retained scalar geometry from ignored gradient vectors."""
import argparse
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def verify(directory,output):
    if output.exists():
        raise ValueError('Fresh scalar verification required')
    result=read(directory/'results.json');config=result['config']
    execution=read(directory/'reports/execution.json')
    assert execution==result['execution'] and execution['status']=='completed'
    assert execution['source_sha256']==sha(HERE/'run.py')
    assert execution['config_sha256']==sha(HERE/'config.json')
    assert execution['all_owned_children_reaped'] and execution['elapsed_s']<=config['wall_seconds']
    assert execution['minimum_host_available_bytes']>=config['host_reserve_bytes']
    assert all(w['returncode']==0 and w['reaped'] and w['peak_rss_bytes']<=config['worker_rss_limit_bytes']for w in execution['workers'])
    assert result['device']=='cuda:0' and result['network_dtype']=='float32'
    assert not result['amp'] and not result['tf32'] and result['deterministic_algorithms']
    assert result['optimizer_updates']==0 and not result['model_promoted'] and result['baseline_retained']=='C53'
    assert result['gpu_peak_allocated_bytes']<=config['gpu_limit_bytes']
    assert result['vector_disk_bytes']<=config['vector_disk_limit_bytes']
    assert result['vector_workspace_bound_bytes']<=config['vector_workspace_limit_bytes']
    assert result['graph_calls']==144 and result['maximum_live_forward_graphs']==1
    assert [s['update']for s in result['snapshots']]==[100,150,200]
    assert config['target_relative_residual']==1e-6
    for sources in (result['sources'],result['summary_sources']):
        for name,identity in sources.items():
            assert sha(ROOT/name)==identity,name
    custody=read(ROOT/config['study']/'receipt.json')
    normalization=ROOT/config['study']/'restore/normalization.json'
    assert sha(normalization)==custody['restore_receipt']['outputs']['normalization.json']['sha256']
    def compare(vector,other,observed):
        # Separate scalar accumulation from the subject's BLAS dot implementation.
        a=vector.astype('float64');b=other.astype('float64')
        dot=float(np.sum(a*b,dtype='float64'))
        an=float(np.sqrt(np.sum(a*a,dtype='float64')));bn=float(np.sqrt(np.sum(b*b,dtype='float64')))
        assert math.isclose(dot,observed['dot'],rel_tol=1e-10,abs_tol=1e-9)
        assert math.isclose(an,observed['left_l2'],rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(bn,observed['right_l2'],rel_tol=1e-12,abs_tol=1e-12)
        if an and bn:
            assert math.isclose(dot/an/bn,observed['cosine'],rel_tol=1e-10,abs_tol=1e-12)
        else:
            assert observed['cosine']is None
    disk=0;opposed=0
    for snapshot in result['snapshots']:
        replay=snapshot['cuda_replay']
        assert replay['raw_exact'] and replay['clipped_exact'] and replay['maximum_loss_relative_gap']==0
        assert replay['raw_relative_l2_gap']==replay['clipped_relative_l2_gap']==0
        assert snapshot['network_unchanged']
        update=snapshot['update'];weak=config['weak_case'];vectors={}
        assert {(r['case'],r['term'])for r in snapshot['rows']}=={(i,t)for i in range(8)for t in ('full','1','2','3','4')}
        for row in snapshot['rows']:
            path=ROOT/row['vector'];assert sha(path)==row['sha256']
            vectors[row['case'],row['term']]=np.load(path,allow_pickle=False)
            assert vectors[row['case'],row['term']].dtype==np.float32
            disk+=path.stat().st_size
        with np.load(ROOT/config['study']/'restore'/('adam-trace-%03d.npz'%update),allow_pickle=False)as archive:
            prefix='u%03d_'%update
            raw=archive[prefix+'gradient_raw'];before=archive[prefix+'parameters_before'];after=archive[prefix+'parameters_after']
            delta=after.astype('float64')-before
        assert snapshot['observed_parameter_sha256']==__import__('hashlib').sha256(before.tobytes()).hexdigest()
        for row in snapshot['rows']:
            vector=vectors[row['case'],row['term']]
            assert vector.shape==(result['parameter_count'],)and np.isfinite(vector).all()
            assert math.isclose(row['gradient_l2'],float(np.sqrt(np.sum(vector.astype('float64')**2))),rel_tol=1e-12)
            compare(vector,vectors[weak,'4'],row['vs_weak_final'])
            compare(vector,vectors[weak,'full'],row['vs_weak_full'])
            compare(vector,raw,row['vs_captured_mean'])
            compare(vector,delta,row['vs_observed_delta'])
            if row['term']!='full':
                assert math.isclose(row['loss'],math.log1p(row['relative_residual']**2/1e-12),rel_tol=1e-12)
            else:
                assert row['loss']==replay['losses'][row['case']]
        for pair in snapshot['layer_comparisons']:
            a,b=vectors[weak,'4'],vectors[pair['case'],str(pair['passes'])]
            cursor=0
            assert len(pair['layers'])==len(result['layout'])
            for layer,entry in zip(pair['layers'],result['layout'],strict=True):
                end=cursor+entry['count'];assert layer['name']==entry['name']
                compare(a[cursor:end],b[cursor:end],layer);cursor=end
            assert cursor==result['parameter_count']
            actual=sum(r['dot']for r in pair['layers'])
            assert math.isclose(actual,float(a.astype('float64')@b.astype('float64')),rel_tol=1e-10,abs_tol=1e-8)
            opposed+=actual<0
    assert disk==result['vector_disk_bytes']and opposed==18
    qualification=read(directory/'reports/qualification.json')
    assert qualification['status']=='passed'and qualification['geometry_sha256']==sha(HERE/'geometry.py')
    browser=read(directory/'reports/browser.json')
    assert browser['status']=='passed'and browser['browser_reaped']and not browser['console_errors']
    assert browser['html_sha256']==sha(directory/'index.html')
    cancelled=read(directory/'reports/cancellation.json')
    assert cancelled['status']=='passed'and len(cancelled['tests'])==2
    for test in cancelled['tests']:
        assert test['controller_reaped']and test['execution']['status']=='cancelled'
        assert test['execution']['all_owned_children_reaped']
        assert test['execution']['source_sha256']==execution['source_sha256']
    assert 100 in cancelled['tests'][1]['registered']['completed_updates']
    failed=read(directory/'reports/attempt-01-execution.json')
    assert failed['status']=='failed'and failed['all_owned_children_reaped']
    for name in ('plan.md','execution-ledger.md','commands.md','next-steps.md'):
        assert (directory/name).is_file()
    write(output,dict(status='passed',opposing_selected_pairs=opposed,gradient_vectors_checked=120,
                      exact_cuda_gradient_replay_states=3,all_sources_match=True,
                      all_owned_children_reaped=True,all_layer_dots_recomputed=True,
                      results_sha256=sha(directory/'results.json'),source_sha256=sha(__file__),
                      scope='Frozen CUDA derivative measurements and retained scalar consistency; no finite-step, new-training or rollout qualification.'))
    print('Verified CUDA gradient receipts, 120 vectors, 18 layer partitions, resources and lifecycle',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    verify(args.directory.resolve(),args.output.resolve())
