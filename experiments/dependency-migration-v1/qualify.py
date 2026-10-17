"""Independent, fail-closed migration witnesses; never trains or selects a model.

Admission is standard-library-only. Numerical commands require the retained
numerical Python and a separately verified complete runtime inventory. Historical
sources are read from their bound commit, never from mutable working copies.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_bytes(root, item):
    if 'commit' in item:
        value = subprocess.check_output(['git', '-C', str(root), 'show', item['commit']+':'+item['path']])
    else:
        value = (root/item['path']).read_bytes()
    require(len(value) == item['bytes'] and sha(value) == item['sha256'], 'changed bound asset: '+item['path'])
    return value


def admission(root, bindings):
    """Verify every fixed binding, captured model source closure and all 74 scenes."""
    observed = []
    def visit(value):
        if isinstance(value, dict):
            if {'path', 'bytes', 'sha256'} <= value.keys():
                bound_bytes(root, value)
                observed.append({k:value[k] for k in ('path','bytes','sha256')})
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(bindings)
    reg = bindings['regression']
    witness = json.loads(bound_bytes(root, reg['source_witness']))
    source_rows = []
    for name, expected in witness['sources'].items():
        path = root/reg['project']/name
        actual = digest(path)
        require(actual == expected, 'captured model source changed: '+name)
        source_rows.append({'path':str(path), 'sha256':actual})
    config = json.loads(bound_bytes(root, reg['membership_source']))
    scenes = config['scenes']
    require(len(scenes) == reg['scene_count'] == 74, 'exact 74 scenes required')
    require(sha(canonical(scenes)) == reg['ordered_scenes_sha256'], 'ordered scene identity mismatch')
    for index, (scene, case) in enumerate(zip(scenes, reg['cases'])):
        require(case['scene'] == index and sha(canonical(scene)) == case['scene_sha256'], 'case scene mismatch')
        require(scene['warmup_steps'] == 15 and scene['steps']-15 == case['expected_ai_steps'], 'actual native prefix/horizon mismatch')
    require([sum(c['group'] == g for c in reg['cases']) for g in reg['groups']] == [58,8,8], 'fixed groups mismatch')
    require(json.loads(bound_bytes(root, reg['stopping_policy'])) == reg['stopping_values'], 'stopping policy mismatch')
    return {'passed':True, 'assets':observed, 'captured_sources':source_rows,
            'ordered_scenes_sha256':sha(canonical(scenes)), 'scenes':scenes,
            'scope':'byte/source/scene admission only; no runtime/model/simulation acceptance'}


def case_metrics(directory, scene, binding, stopping, binary_sha, require_sparse=True):
    """Check retained raw evidence independently of a controller's success flag."""
    directory = Path(directory)
    receipt = read(directory/'native/receipt.json')
    frames = [json.loads(s) for s in (directory/'native/frames.jsonl').read_text().splitlines()]
    callbacks = [json.loads(s) for s in (directory/'native/callbacks.jsonl').read_text().splitlines()]
    require(receipt['scene'] == scene and receipt['binary_sha256'] == binary_sha, 'scene/binary mismatch')
    result = receipt['result']
    require(receipt['status'] == 'completed' and receipt['error'] is None and result and not result['hard_stop'], 'incomplete or rejected native child')
    require(receipt['native_process']['reaped'] and receipt['native_process']['returncode'] == 0, 'unreaped/failed child')
    require(receipt['cpu_threads'] == 1 and receipt['cuda_initialized'] is False, 'one-thread CPU required')
    require(receipt['native_pressure_refinement'] is False and not receipt.get('native_pressure_fallback',False) and not receipt.get('native_fallback',False), 'native refinement/fallback forbidden')
    require(receipt['adaptive_stopping'] == stopping and receipt['proposal_network_evaluations'] == 1
            and receipt['learned_proposals_per_pressure_solve'] == 32 and receipt['minimum_network_evaluations'] == 0, 'changed inference policy')
    require(receipt.get('rejected_attempts',0)==0, 'rejected attempt cannot qualify')
    expected = list(range(16, scene['steps']+1))
    ai = [f for f in frames if f.get('ai_quality')]
    require([f['step'] for f in frames] == list(range(1,scene['steps']+1)), 'missing/reordered physical frames')
    require([f['step'] for f in ai] == [c['step'] for c in callbacks] == expected, 'missing/reordered AI callbacks')
    require(result['ai_steps'] == result['qualified_ai_steps'] == len(expected) == binding['expected_ai_steps'], 'incomplete AI horizon')
    for f in frames:
        q=f['reference_quality']
        require(q['qualified'] and not q['hard_stop'] and q['actual_divergence'] <= 1e-5 and q['courant'] <= .5, 'native physical gate failure')
    calls = []
    for f,c in zip(ai,callbacks):
        q=f['ai_quality']; last=c['trace'][-1]
        require(q['qualified'] and not q['hard_stop'] and q['actual_divergence'] <= 1e-5 and q['courant'] <= .5, 'AI physical gate failure')
        require(not f.get('native_fallback',False) and not f.get('native_refinement',False), 'fallback/refinement is not AI solving')
        diagnostic=f.get('diagnostic_attempt')
        if diagnostic is not None:
            outcome=diagnostic['outcome']
            require(diagnostic['dispatched'] and diagnostic['generation']==f['step']
                    and diagnostic['native_pressure_refinement'] is False and diagnostic['native_fallback'] is False, 'diagnostic dispatch/history/fallback violation')
            require(outcome['kind']=='candidate' and outcome['qualified'] and not outcome['hard_stop']
                    and outcome['walls_closed'] and abs(outcome['net_flux'])<=outcome['flux_limit'], 'candidate wall/flux/acceptance violation')
            require(outcome['relative_limit']==1e-6 and outcome['predicted_limit']==1e-7
                    and outcome['actual_limit']==1e-5 and outcome['courant_limit']==.5, 'native physical limits changed')
        n=c['network_passes']
        require(type(n) is int and 0 <= n <= 32 and c['proposals'] == n, 'network call policy changed')
        require(last['equation_passed'] and last['pressure_error_bound_passed'], 'conjunctive stopping failed')
        require(last['residual_l2'] <= max(1e-12,1e-6*last['rhs_l2']) and last['predicted_divergence'] <= 1e-7
                and last['mean_free_pressure_relative_error_bound'] <= 1e-6, 'relaxed stopping criteria')
        calls.append(n)
    if require_sparse:
        samples=read(directory/'diagnostics.json')['samples']
        wanted=sorted({16,scene['steps'],*range(64,scene['steps']+1,64)})
        require([s['step'] for s in samples] == wanted, 'missing sparse same-RHS oracle coverage')
        require(all(s['oracle_independent_relative_defect'] < 1e-9 and s['pressure_bound_covers_oracle'] for s in samples), 'independent operator/bound failure')
    return {'scene':binding['scene'], 'group':binding['group'], 'qualified':True, 'complete_window':True,
            'calls':calls, 'calls_sha256':sha(canonical(calls)), 'mean_calls':statistics.mean(calls), 'maximum_calls':max(calls),
            'worst_paired_pressure_relative_l2':max(f['relative_pressure_difference'] for f in ai),
            'receipt_sha256':digest(directory/'native/receipt.json')}


def accept_case(archived, old, new=None):
    """No numerical tolerance, including archived/reproduced pressure maxima."""
    require(old['calls_sha256'] == archived['calls_sha256'], 'old call-sequence reproduction failed')
    require(old['worst_paired_pressure_relative_l2'] == archived['worst_paired_pressure_relative_l2'], 'old pressure-error reproduction failed')
    require(old['mean_calls'] == archived['mean_calls'] and old['maximum_calls'] == archived['maximum_calls'], 'old call-summary reproduction failed')
    if new is not None:
        require(new['worst_paired_pressure_relative_l2'] <= min(archived['worst_paired_pressure_relative_l2'], old['worst_paired_pressure_relative_l2']), 'new per-case pressure regression')
        require(new['mean_calls'] <= old['mean_calls'] and new['maximum_calls'] <= old['maximum_calls'], 'new per-case network call regression')
    return True


def native_metrics(directory, scene, binary_sha):
    directory=Path(directory);receipt=read(directory/'receipt.json')
    frames=[json.loads(s) for s in (directory/'frames.jsonl').read_text().splitlines()]
    require(receipt['scene']==scene and receipt['binary_sha256']==binary_sha,'native control scene/binary mismatch')
    require(receipt['status']=='completed' and receipt['native_process']['reaped'] and receipt['native_process']['returncode']==0
            and receipt['result']['reference_control'] and not receipt['result']['hard_stop'],'native control failed/rejected/unreaped')
    require([f['step'] for f in frames]==list(range(1,scene['steps']+1)),'native control incomplete horizon')
    require([f['step'] for f in frames if f.get('ai_quality')]==list(range(16,scene['steps']+1)),'native control AI comparison/prefix coverage')
    require(all(f['reference_quality']['qualified'] and (f.get('ai_quality') is None or f['ai_quality']['qualified']) for f in frames),'native control physical failure')
    require(receipt['cpu_threads']==1 and receipt['cuda_initialized'] is False,'native control runtime mismatch')
    return {'qualified':True,'complete_window':True,'receipt_sha256':digest(directory/'receipt.json')}


def compare_cohorts(bindings, old, new=None, native=None):
    expected=bindings['regression']['cases']
    require([r['scene'] for r in old] == list(range(74)), 'complete ordered old cohort required')
    if new is not None:
        require([r['scene'] for r in new] == list(range(74)), 'complete ordered new cohort required')
        require(native is not None and [r['scene'] for r in native]==list(range(74))
                and all(r['qualified'] and r['complete_window'] for r in native),'74 matched qualified new native controls required')
    for i,a in enumerate(expected):
        require(old[i]['qualified'] and old[i]['complete_window'] and old[i]['group']==a['group'], 'old incomplete/group mismatch')
        if new is not None:
            require(new[i]['qualified'] and new[i]['complete_window'] and new[i]['group']==a['group'], 'new incomplete/group mismatch')
        accept_case(a, old[i], None if new is None else new[i])
    groups={}
    for g in bindings['regression']['groups']:
        def describe(rows):
            rows=[r for r in rows if r['group']==g]; calls=[n for r in rows for n in r['calls']]
            return {'mean_calls':statistics.mean(calls),'maximum_calls':max(calls),'worst_paired_pressure_relative_l2':max(r['worst_paired_pressure_relative_l2'] for r in rows)}
        groups[g]={'old':describe(old)}
        if new is not None:
            groups[g]['new']=describe(new)
            require(all(groups[g]['new'][k] <= groups[g]['old'][k] for k in groups[g]['old']), 'group regression')
    return {'passed':True,'groups':groups,'cases':74,'same_rhs_and_trajectory_metrics_distinct':True}


def load_old(root, bindings):
    source_root=Path(__file__).resolve().parents[2]
    for relative in ('experiments/water-residual-learning-v1/audit.py','experiments/water-stability-learning-v1/learned_iteration.py','experiments/global-pressure-v1/verify_capture.py'):
        expected=subprocess.check_output(['git','-C',str(root),'show',bindings['source_commit']+':'+relative])
        require((source_root/relative).read_bytes() == expected, 'changed baseline reference source: '+relative)
    sys.path.insert(0,str(source_root/'experiments/water-residual-learning-v1'))
    import audit
    reg=bindings['regression']
    audit.initialize(root/reg['project'],root/reg['artifacts'],sorted(os.sched_getaffinity(0)))
    sys.path.insert(0,str(source_root/'experiments/water-stability-learning-v1'))
    import learned_iteration
    return audit,learned_iteration


def frame_reference(root, bindings, output):
    import numpy as np
    audit,iteration=load_old(root,bindings); torch=audit.TORCH; reg=bindings['regression']
    with np.load(root/reg['first_frame']['capture']['path'],allow_pickle=False) as arrays:
        sample={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files if k!='prediction'}
        retained=torch.from_numpy(arrays['prediction'].copy())
    require(sample['base'].shape == (1,1,9,13,17) and float(sample['cell_size_m'][0]) == .125
            and float(sample['rho'][0]) == 1000. and float(sample['dt'][0]) == 1/60, 'fixed first-frame context mismatch')
    def arraysha(t):
        a=t.detach().numpy();return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':sha(a.tobytes(order='C'))}
    trace=[]
    with torch.no_grad():
        inputs=audit.MODEL.build_inputs(sample['native_input_residual'],sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],audit.NORM)
        raw=audit.NETWORK(inputs)
        prediction=iteration.correct(audit.MODEL,audit.NETWORK,audit.NORM,sample,32,reg['stopping_values'],trace,True,0)
        zero={**sample,'native_input_residual':torch.zeros_like(sample['native_input_residual'])}
        bypass=audit.MODEL.predict_pressure(audit.NETWORK,zero,audit.NORM)
        require(torch.equal(prediction,retained), 'old saved first-frame pressure does not reproduce exactly')
        require(torch.equal(bypass,sample['base']), 'zero residual bypass changed base')
    from importlib.util import spec_from_file_location,module_from_spec
    source_root=Path(__file__).resolve().parents[2]
    spec=spec_from_file_location('migration_neighbor_oracle',source_root/'experiments/global-pressure-v1/verify_capture.py')
    oracle=module_from_spec(spec);spec.loader.exec_module(oracle)
    def independent_action(delta):
        values=oracle.full_action(delta.flatten().tolist(),list(delta.shape[2:][::-1]),[float(sample['cell_size_m'][0])]*3,float(sample['rho'][0]))
        return torch.from_numpy(np.asarray(values,dtype='<f8')).reshape_as(delta)
    delta=audit.spectral_correction(sample['native_input_residual'],sample)
    compatible=sample['native_input_residual']-sample['native_input_residual'].mean()
    defect=audit.vector_norm(independent_action(delta)-compatible)/audit.vector_norm(compatible)
    require(defect < 1e-9 and float(delta.flatten()[0])==0, 'independent same-RHS operator/gauge mismatch')
    two=torch.tensor([[[[[0.,2.]]]]],dtype=torch.float64)
    acted=audit.MODEL.full_operator(two,torch.tensor([.5],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))
    require(torch.allclose(acted,torch.tensor([[[[[-.001,.001]]]]],dtype=torch.float64),rtol=0,atol=1e-15), 'analytical two-cell stencil mismatch')
    values={'input_float32':inputs.numpy(),'network_output_float32':raw.numpy(),'full_pressure_float64':prediction.numpy(),'zero_bypass_float64':bypass.numpy()}
    with (output/'reference-arrays.npz').open('xb') as stream:np.savez(stream,**values)
    return {'passed':True,'same_input_exact_to_retained':True,'zero_residual_exact':True,'optimizer_updates':0,
            'cuda_initialized':torch.cuda.is_initialized(),'torch_threads':torch.get_num_threads(),
            'array_witnesses':{k:arraysha(torch.from_numpy(v)) for k,v in values.items()},
            'native_input_array_witnesses':{k:arraysha(v) for k,v in sample.items()},
            'trace':trace,'independent_same_rhs_relative_defect':defect,'two_cell_stencil_passed':True,
            'scope':'old-side fixed first-frame reproduction only; new adapter parity pending'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('admission','old-frame','compare'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--old',type=Path); p.add_argument('--new',type=Path);p.add_argument('--native',type=Path)
    args=p.parse_args(); started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=False)
    receipt={'command':sys.argv,'source_sha256':digest(__file__),'bindings_sha256':digest(args.bindings),
             'python':sys.executable,'python_version':sys.version,'cpu_affinity':sorted(os.sched_getaffinity(0)),
             'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE')},'optimizer_updates':0}
    try:
        b=read(args.bindings)
        if args.mode=='admission': result=admission(args.root,b)
        elif args.mode=='old-frame': result=frame_reference(args.root,b,args.output)
        else: result=compare_cohorts(b,read(args.old),read(args.new) if args.new else None,read(args.native) if args.native else None)
        receipt.update(status='passed',result=result)
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (args.output/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds']}))


if __name__=='__main__':main()
