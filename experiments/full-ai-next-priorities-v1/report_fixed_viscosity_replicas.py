"""Audit added-seed managed fits and full accuracy without reselecting the primary."""
import statistics
import numpy as np
from summarize_coupled import aggregate,FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'fixed-viscosity-replication-results.json'
    if output.exists():raise ValueError('Fresh family report required')
    batch_path=BASE/'fixed-viscosity-replication-training-batch-01/execution.json';batch=read(batch_path)
    if batch['status']!='running'or batch.get('active',{}).get('id')!='audit-six-managed-replicas-and-accuracy':
        raise ValueError('Reporter must own the final read-only batch command')
    specification_path=DOCS/'fixed-viscosity-replication-training-batch-specification-01.json'
    specification=read(specification_path)
    if [c['id']for c in batch['completed']]!=[c['id']for c in specification['commands'][:-1]]or not all(c['reaped']and c['returncode']==0 for c in batch['completed']):
        raise ValueError('All six finite fits/restores/evaluations must complete and reap')
    reg_path=DOCS/'fixed-viscosity-replication-fit-registration-01.json';reg=read(reg_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered family source changed')
    registry_path=BASE/'fixed-viscosity-replication-inputs-01/models.json';registry=read(registry_path)
    sources=[reg_path,registry_path,DOCS/'fixed-viscosity-family-plan.md',DOCS/'fixed-viscosity-replication-selection-01.json',DOCS/'fixed-viscosity-family-results.json']
    preceding_commands=batch['completed']
    sources.extend([specification_path,DOCS/'fixed-viscosity-replication-batch-registration-01.json'])
    for command in preceding_commands:
        log=ROOT/command['log']
        if sha(log)!=command['log_sha256']:raise ValueError('Completed command log changed')
        sources.append(log)
    common_initial={};common_static=None;common_config={};rows=[];checks={};work_rows=[]
    for entry in registry['models']:
        k=entry['horizons'];ordinal=entry['replication_ordinal'];identifier=entry['identifier']
        pair=BASE/(identifier+'-pair-01');receipt=read(pair/'receipt.json')
        project=ROOT/receipt['source_project'];assembly=read(project/'assembly.json');config=assembly['config']
        if sha(project/'assembly.json')!=receipt['source_assembly_sha256']:
            raise ValueError('Captured Project identity changed')
        for source in assembly['sources']:
            if sha(ROOT/source['path'])!=source['sha256']:raise ValueError('Captured source/assets changed')
        trained=read(pair/'train/receipt.json');restored=read(pair/'restore/receipt.json')
        history=read(pair/'restore/history.json');initial=history['initial_inventory'];final=history['final_inventory']
        work=history['optimizer_work'];forwards=2*config['updates']-config['single_block_updates']
        values={name:config[name]for name in ('learning_rate','seed','sampling_seed','physical_scales','supervision_mode',
            'feedback_protocol','step_scale_statistics_sha256','original_normalization_sha256','displacement_auxiliary_weight')}
        if ordinal not in common_initial:common_initial[ordinal]=initial;common_config[ordinal]=values
        if common_static is None:common_static=assembly['static_inputs']
        local=dict(exact_CPU_artifact_parity=all(receipt['artifact_parity'].values()),
            actual_registered_updates=trained['optimizer_updates']==config['updates'],
            zero_update_CPU_restore=restored['optimizer_updates']==0 and restored['full_cpu_probe_bitwise_verified'],
            typed_original_Run=restored['originating_training_run']['run_id']==receipt['training_run_id'],
            same_captured_closure=trained['closure']==restored['closure'],
            actual_cuda=trained['execution']['device']=='cuda'and trained['execution']['cuda_initialized'],
            CPU_cuda_not_initialized=not restored['execution']['cuda_initialized'],
            common_initial_inventory=initial==common_initial[ordinal],common_static_inputs=assembly['static_inputs']==common_static,
            common_registered_controls=values==common_config[ordinal],registered_config=sha(project/'config.json')==reg['config_sha256'][identifier],
            actual_registered_seeds=config['seed']==entry['initialization_seed']and config['sampling_seed']==entry['sampling_seed'],
            frozen_coefficients_unchanged=all(initial[n]==final[n]==spec for n,spec in trained['frozen_parameter_inventory'].items()),
            viscosity_frozen_from_start='viscous_head.weight'in trained['frozen_parameter_inventory'],
            exact_trainable_inventory=trained['trainable_parameters']==6600,
            expected_block_forwards=work['network_forwards']==forwards,one_encoder_per_block=work['encoder_calls']==forwards,
            physical_targets_36000=work['predicted_physical_states']==36000,
            all_pressure_readouts=work['pressure_readout_calls']==36000,
            all_displacement_readouts=work['displacement_readout_calls']==36000,
            deployed_internal_feedback=work['internal_history_updates']==forwards*(k-1))
        with np.load(pair/'restore/weights.npz',allow_pickle=False)as arrays,np.load(project/'expert-assets/viscosity-weights.npz',allow_pickle=False)as component:
            local['exact_managed_viscosity']=np.array_equal(arrays['viscous_head.weight'].view('u1'),component['projection.weight'].view('u1'))
        checks[identifier]=local
        if not all(local.values()):
            write(BASE/'fixed-viscosity-replication-control-audit-failure-01.json',dict(status='failed',checks=checks))
            raise ValueError('Actual family control audit failed')
        path=BASE/'fixed-viscosity-replication-evaluation-01'/identifier/'summary.json';evaluation=read(path)
        wrapper_path=path.parent/'wrapper-receipt.json';wrapper=read(wrapper_path)
        if evaluation['status']!='completed'or len(evaluation['scenes'])!=8 or evaluation['bank_sha256']!=reg['bank_sha256']:
            raise ValueError('All eight unchanged development scenes required')
        if evaluation['model']['weights_sha256']!=sha(pair/'restore/weights.npz'):
            raise ValueError('Evaluated checkpoint differs from managed exact CPU restore')
        if wrapper['core_summary_sha256']!=sha(path)or wrapper['actual_loader_sha256']!=sha(HERE/'fixed_viscosity_evaluation_support.py')or not wrapper['training_teacher_disabled']:
            raise ValueError('Actual new loader and teacher isolation required')
        for scene in evaluation['scenes']:
            raw_path=path.parent/(scene['identifier']+'.json');raw=read(raw_path);sources.append(raw_path)
            single=[r for r in raw['rows']if r['mode']=='single_block']
            if {(r['start'],r['horizon'])for r in single}!={(s,h)for s in range(121)for h in range(1,k+1)}or len(single)!=121*k:
                raise ValueError('Every declared intermediate output required')
            if scene['first_unavailable_step']is None:
                own=[r['step']for r in raw['rows']if r['mode']=='own_history']
                if own!=list(range(1,241)):raise ValueError('Complete own240 physical outputs required')
        for regime in ('small','32cubed'):
            for index,interval in enumerate(('single_block','rollout_12','rollout_48','rollout_240')):
                rows.append(dict(identifier=identifier,horizons=k,replication_ordinal=ordinal,regime=regime,interval=interval,**aggregate(evaluation,regime,index)))
        work_rows.append(dict(identifier=identifier,horizons=k,replication_ordinal=ordinal,optimizer_updates=config['updates'],
            single_block_updates=config['single_block_updates'],optimizer_work=work,setup_seconds=history['setup_seconds'],
            update_seconds_sum=sum(history['per_update_seconds']),median_update_seconds=statistics.median(history['per_update_seconds']),
            training_elapsed_s=trained['elapsed_s'],training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id']))
        sources.extend([pair/'receipt.json',pair/'train/receipt.json',pair/'restore/receipt.json',pair/'restore/history.json',
            pair/'restore/weights.npz',pair/'restore/initial-weights.npz',project/'assembly.json',project/'config.json',path,wrapper_path])
    if set(common_initial)!=set((2,3))or common_initial[2]==common_initial[3]:raise ValueError('Actual distinct additional initializations required')
    passed=[m['identifier']for m in registry['models']if all(r['accuracy_pass']for r in rows if r['identifier']==m['identifier'])]
    write(output,dict(status='completed',rows=rows,control_audit=checks,training_work=work_rows,
        accuracy_qualified_all_eight_scenes=passed,all_intermediate_outputs_required=True,own_history_steps=240,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in sources},reporter_sha256=sha(__file__),
        comparison='Two added temporal-cell seeds per selected bundle and B1; shared frozen learned components and36000physical targets per fit',
        preceding_completed_commands=preceding_commands,primary_seed_reselected=False,independent_component_pretraining=False,
        matched_FLOPs_claim=False,single_factor_claim_against_old_composition=False,
        CPU_latency_claim=False,replicas_pending=False,execution_precision_and_replica_cost_pending=True,reserved_scenes_consumed=False,original_goal_complete=False))
    lines=['# Fixed neural viscosity: two additional seeds per selected recipe','',
        'All six additional-seed managed CUDA fits, exact CPU restores and development evaluations complete. '
        'The same 243 learned viscosity coefficients are frozen from initialization. Every model supervises 36,000 physical states; '
        'different optimizer counts and internal work are reported, with no equal-FLOP claim.','',
        '| Model | Grid group | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        text=' / '.join(f'{row["worst"][f]:.7g}'for f in FIELDS)if row['complete']else'Unavailable; incomplete interval'
        lines.append(f'| {row["identifier"]} | {row["regime"]} | {row["interval"]} | {row["complete"]} | {text} | {row["accuracy_pass"]} |')
    lines.extend(['','Every horizon and all 240 own-history steps use the original equation/field and near-zero criteria. '
        'These are accuracy and custody results; actual replica CPU speed and qualified execution variants, '
        'reserved trajectories and sustained application/event costs remain separate.','',
        '[Exact work, controls, raw evidence and hashes](fixed-viscosity-family-results.json) · '
        '[Registered family](fixed-viscosity-family-plan.md)',''])
    (DOCS/'fixed-viscosity-replication-results.md').write_text('\n'.join(lines))
    print('Audited all six managed replicas; accuracy-qualified all8scenes:',passed,flush=True)


if __name__=='__main__':run()
