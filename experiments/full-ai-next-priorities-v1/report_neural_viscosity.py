"""Publish all-neural viscosity repair against unchanged every-output criteria."""
import numpy as np
from summarize_coupled import aggregate,FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'neural-viscosity-results.json'
    if output.exists():raise ValueError('Fresh repaired-model result required')
    directory=BASE/'b2-displacement-viscosity-assembly-01';assembly=read(directory/'receipt.json')
    parent=ROOT/assembly['parent_pair'];component=ROOT/assembly['component_pair']
    reg_path=DOCS/'neural-viscosity-composition-registration-01.json';reg=read(reg_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered neural inference/evaluation source changed')
    if assembly['changed_tensors']!=['viscous_head.weight']or not assembly['all_other_tensors_bitwise_unchanged']or assembly['optimizer_updates']!=0:
        raise ValueError('Only243 learned viscosity coefficients may change')
    parent_receipt=read(parent/'receipt.json');component_receipt=read(component/'receipt.json')
    if not all(parent_receipt['artifact_parity'].values())or not all(component_receipt['artifact_parity'].values()):
        raise ValueError('Both managed parents require exact CPU restoration')
    with np.load(directory/'weights.npz',allow_pickle=False)as arrays,np.load(parent/'restore/weights.npz',allow_pickle=False)as old,np.load(component/'restore/weights.npz',allow_pickle=False)as learned:
        if set(arrays.files)!=set(old.files):raise ValueError('Full neural inventory changed')
        changed=[n for n in arrays.files if not np.array_equal(arrays[n].view('u1'),old[n].view('u1'))]
        if changed!=['viscous_head.weight']or not np.array_equal(arrays['viscous_head.weight'].view('u1'),learned['projection.weight'].view('u1')):
            raise ValueError('Composed model differs from exact managed sources')
    prior_path=BASE/'temporal-displacement-evaluation-01/b2-displacement/summary.json'
    path=BASE/'neural-viscosity-evaluation-01/b2-displacement-viscosity/summary.json';value=read(path);prior=read(prior_path)
    wrapper_path=path.parent/'wrapper-receipt.json';wrapper=read(wrapper_path)
    batch=read(BASE/'neural-viscosity-evaluation-batch-01/execution.json')
    if batch['status']!='completed'or not batch['owned_child_reaped']or value['status']!='completed'or len(value['scenes'])!=8:
        raise ValueError('Full development batch and reaping required')
    if value['bank_sha256']!=prior['bank_sha256']or value['metrics_sha256']!=prior['metrics_sha256']or value['bank_sha256']!=reg['bank_sha256']:
        raise ValueError('Evaluation scenes or original thresholds changed')
    if wrapper['core_summary_sha256']!=sha(path)or wrapper['actual_loader_sha256']!=sha(HERE/'viscosity_composition_support.py')or not wrapper['training_teacher_disabled']:
        raise ValueError('Actual new loader and inference teacher isolation required')
    if value['model']['weights_sha256']!=assembly['weights_sha256']or sha(directory/'weights.npz')!=assembly['weights_sha256']:
        raise ValueError('Actual evaluated weights differ')
    raw=[]
    for scene in value['scenes']:
        source=path.parent/(scene['identifier']+'.json');record=read(source);raw.append(source)
        one=[r for r in record['rows']if r['mode']=='single_block'];own=[r for r in record['rows']if r['mode']=='own_history']
        if len(one)!=242 or {(r['start'],r['horizon'])for r in one}!={(s,h)for s in range(121)for h in (1,2)}:
            raise ValueError('Every reference-history output required')
        if scene['first_unavailable_step']is None and [r['step']for r in own]!=list(range(1,241)):
            raise ValueError('Every own-history output required')
    comparisons=[]
    for regime in ('small','32cubed'):
        for i,label in enumerate(('single_block','own_history12','own_history48','own_history240')):
            comparisons.append(dict(regime=regime,interval=label,parent=aggregate(prior,regime,i),neural_viscosity=aggregate(value,regime,i)))
    passed=all(r['neural_viscosity']['accuracy_pass']for r in comparisons)
    sources=[path,prior_path,wrapper_path,*raw,directory/'receipt.json',directory/'weights.npz',reg_path,
        parent/'receipt.json',component/'receipt.json',BASE/'viscosity-calibration-native-evaluation-01.json',
        BASE/'neural-viscosity-evaluation-batch-01/execution.json',DOCS/'viscosity-calibration-results.json']
    write(output,dict(status='completed',identifier=assembly['identifier'],comparisons=comparisons,parameters=100456,
        changed_parameters=243,all_other_neural_tensors_exact=True,accuracy_qualified_all_eight_scenes=passed,
        accuracy_qualified_small_grid_recipe=all(r['neural_viscosity']['accuracy_pass']for r in comparisons if r['regime']=='small'),
        managed_parent_full_state_updates=10000,managed_component_updates=1000,assembly_optimizer_updates=0,
        numerical_viscosity_in_inference=False,source_evidence={str(p.relative_to(ROOT)):sha(p)for p in sources},reporter_sha256=sha(__file__),
        CPU_latency_claim=False,combined_target_claim=False,K3_K4_and_replicas_unproven=True,reserved_scenes_consumed=False,original_goal_complete=False))
    lines=['# Neural viscosity repair: full-state accuracy','',
        'Only243 viscosity coefficients change. They come from a completed managed neural fit and exact CPU restore. '
        'All other displacement, encoder, pressure and flux tensors are byte-identical to the parent. '
        'The assembled model calls no native viscosity primitive during inference.','',
        '| Regime | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |',
        '| --- | --- | --- | --- | --- |']
    for row in comparisons:
        arm=row['neural_viscosity'];text=' / '.join(f'{arm["worst"][f]:.7g}'for f in FIELDS)if arm['complete']else'Unavailable; incomplete'
        lines.append(f'| {row["regime"]} | {row["interval"]} | {arm["complete"]} | {text} | {arm["accuracy_pass"]} |')
    lines.extend(['','Accuracy uses every intermediate field on all eight development scenes and through240own-history steps. '
        'The original normalization, near-zero rules and0.004gate remain unchanged. CPU speed, K3/K4, seed replication and reserved trajectories '
        'remain separate gates; this accuracy report alone does not qualify the full objective.','',
        '[Complete evaluation and provenance](neural-viscosity-results.json) · '
        '[Managed neural component](viscosity-calibration-results.md) · [Original plan](plan.md)',''])
    (DOCS/'neural-viscosity-results.md').write_text('\n'.join(lines))
    print('Published repaired neural model accuracy; all eight scenes/all intervals pass:',passed,flush=True)


if __name__=='__main__':run()
