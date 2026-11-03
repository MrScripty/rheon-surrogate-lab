"""Select two development-qualified recipes without selecting a different primary seed."""
from study_support import ROOT,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-replication-selection-01.json'
    if output.exists():raise ValueError('Fresh concrete replication selection required')
    batch_path=BASE/'fixed-viscosity-k2-precision-publication-batch-01/execution.json';batch=read(batch_path)
    if batch['status']!='completed'or not batch['owned_child_reaped']or batch.get('active')is not None:
        raise ValueError('Dependent read-only precision publisher must finish and reap before selection')
    original_path=BASE/'fixed-viscosity-k2-precision-timing-batch-01/execution.json';original=read(original_path)
    repair_path=DOCS/'fixed-viscosity-k2-precision-publication-repair-01.json';repair=read(repair_path)
    if repair['original_batch_sha256']!=sha(original_path)or repair['numerical_training_or_timing_replayed']:
        raise ValueError('Exact retained publication-only failure must be registered without numerical replay')
    if original['status']!='stopped'or not original['owned_child_reaped']or original['completed'][0]['returncode']!=0:
        raise ValueError('Original successful timing owner must be terminal and reaped')
    family_path=DOCS/'fixed-viscosity-cost-results.json';family=read(family_path)
    precision_path=DOCS/'fixed-viscosity-k2-precision-results.json';precision=read(precision_path)
    paths=[family_path,precision_path,batch_path,original_path,repair_path,DOCS/'fixed-viscosity-family-results.json']
    for report in (family,precision):
        if report['status']!='completed':raise ValueError('Actual completed cost/accuracy evidence required')
        for relative,digest in report['source_evidence'].items():
            if sha(ROOT/relative)!=digest:raise ValueError('Selection evidence changed')
    def point(rows,k,policy=None):
        return next(p for p in rows if p['horizons']==k and p['regime']=='small'and p['interval']=='rollout_240'
                    and(policy is None or p['policy']==policy))
    fourth=point(family['points'],4);third=point(family['points'],3)
    second=point(precision['points'],2,'fast-float32-map')
    # Prefer the shortest passing block plus the longest sharing horizon. If K2
    # fails the actual new-weight comparison, replicate the already-passing K3.
    chosen=(2,4)if second['target_met']else(3,4)
    if not fourth['target_met']or(chosen[0]==3 and not third['target_met']):
        raise ValueError('Both selected recipes must actually meet development accuracy and speed')
    if chosen[0]==2:
        value=read(BASE/'fixed-viscosity-k2-float32-evaluation-01/summary.json')
        if not all(s['all_accuracy_criteria_pass']for scene in value['scenes']for s in scene['summaries']):
            raise ValueError('Actual K2precision must pass all8scenes, not only small-grid aggregate')
    policies={'1':'fast-float64','4':'fast-float64',str(chosen[0]):'fast-float32-map'if chosen[0]==2 else'fast-float64'}
    write(output,dict(status='selected_before_additional_seed_training',recipe_horizons=list(chosen),
        execution_policy_by_horizon=policies,one_step_control_included=True,
        rationale='Shortest qualifying block and longest sharing horizon; K3 retained as measured primary regardless of replication coverage',
        source_evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p))for p in paths],
        primary_seed_policy='Original registered primary seed; replicas characterize variation, not primary reselection',
        original_primary_initialization_seed=2026102045,original_primary_sampling_seed=2026101922,
        added_initialization_seeds=[2026102046,2026102047],added_sampling_seeds=[2026101923,2026101924],
        shared_frozen_components=True,independent_component_pretraining=False,
        all_requested_primary_K_retained=True,reserved_scenes_consumed=False,original_goal_complete=False))
    print('Selected bundled recipes',chosen,'plus matched B1control; original primary seed retained',flush=True)


if __name__=='__main__':run()
