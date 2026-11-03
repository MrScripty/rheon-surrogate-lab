"""Prepare two additional seeds for concretely selected recipes and B1 control."""
import argparse
from pathlib import Path
import shutil
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


SEEDS=((2,2026102046,2026101923),(3,2026102047,2026101924))


def run(args):
    output=BASE/'fixed-viscosity-replication-inputs-01'
    if output.exists():raise ValueError('Fresh replication inputs required')
    selection=read(args.selection);horizons=selection['recipe_horizons']
    if not 1<=len(horizons)<=2 or len(set(horizons))!=len(horizons)or not set(horizons)<={2,3,4}:
        raise ValueError('One or two concrete bundled recipes required')
    if selection['primary_seed_policy']!='Original registered primary seed; replicas characterize variation, not primary reselection':
        raise ValueError('Primary seed must remain fixed before reserved evaluation')
    for entry in selection['source_evidence']:
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Actual recipe-selection evidence changed')
    for name in ('fixed-viscosity-postprocessing-batch-01','float32-map-publication-batch-01'):
        batch=read(BASE/name/'execution.json')
        if batch['status']!='completed'or not batch['owned_child_reaped']:
            raise ValueError('Existing CPU campaigns must complete and reap before replication')
    primary=BASE/'fixed-viscosity-inputs-01';source=read(primary/'receipt.json')
    for name,digest in source['expert_asset_sha256'].items():
        if sha(primary/'expert-assets'/name)!=digest:raise ValueError('Shared frozen learned components changed')
    output.mkdir(parents=True);shutil.copytree(primary/'expert-assets',output/'expert-assets');models=[]
    for ordinal,initialization_seed,sampling_seed in SEEDS:
        for k in (1,*sorted(horizons)):
            project=BASE/f'b{k}-fixed-viscosity-project-01';config=read(project/'config.json')
            identifier=f'b{k}-fixed-viscosity-s{ordinal}'
            config.update(identifier=identifier,seed=initialization_seed,sampling_seed=sampling_seed,
                replication_ordinal=ordinal,replicates_primary=f'b{k}-fixed-viscosity',
                primary_config_sha256=sha(project/'config.json'),replication_selection_sha256=sha(args.selection))
            if (2*config['updates']-config['single_block_updates'])*k!=36000:
                raise ValueError('Original matched physical-target work must remain fixed')
            path=output/(identifier+'-config.json');write(path,config)
            models.append(dict(identifier=identifier,horizons=k,replication_ordinal=ordinal,
                initialization_seed=initialization_seed,sampling_seed=sampling_seed,config_path=path.name,
                config_sha256=sha(path),primary_project=str(project.relative_to(ROOT)),
                primary_weights_sha256=sha(BASE/f'b{k}-fixed-viscosity-pair-01/restore/weights.npz'),
                optimizer_updates=config['updates'],physical_target_states=36000))
    write(output/'models.json',dict(models=models,selected_bundled_horizons=sorted(horizons),
        one_step_control_included=True,shared_frozen_components=True,independent_component_pretraining=False,
        selection_path=str(args.selection.resolve().relative_to(ROOT)),selection_sha256=sha(args.selection)))
    write(output/'receipt.json',dict(status='prepared',source_sha256=sha(__file__),models_sha256=sha(output/'models.json'),
        config_sha256={m['identifier']:m['config_sha256']for m in models},
        expert_asset_sha256=source['expert_asset_sha256'],selection_sha256=sha(args.selection),
        expected_qualification_cases_per_device=len(models)*16,optimizer_updates=0,training_admitted=False,
        reserved_scenes_consumed=False))
    print('Prepared additional temporal-cell seeds and matched B1control:',len(models),'fits; no training admitted',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selection',type=Path,required=True);run(parser.parse_args())
