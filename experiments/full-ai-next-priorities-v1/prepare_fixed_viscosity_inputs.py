"""Prepare bounded K1/2/3/4 fits with fixed neural viscosity from initialization."""
import shutil
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=BASE/'fixed-viscosity-inputs-01'
    if output.exists():raise ValueError('Fresh fixed-viscosity inputs required')
    pair=BASE/'viscosity-calibration-pair-01';receipt=read(pair/'receipt.json')
    evaluation=BASE/'viscosity-calibration-native-evaluation-01.json'
    report=read(DOCS/'viscosity-calibration-results.json')
    if receipt['status']!='completed'or not all(receipt['artifact_parity'].values())or report['status']!='completed_managed_neural_component_calibration':
        raise ValueError('Managed exact-restored qualified viscosity required')
    for name,digest in report['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Viscosity component evidence changed')
    output.mkdir(parents=True);assets=output/'expert-assets'
    prior=BASE/'temporal-displacement-inputs-01';source=read(prior/'receipt.json')
    for name,digest in source['expert_asset_sha256'].items():
        if sha(prior/'expert-assets'/name)!=digest:raise ValueError('Original point/flux/pressure assets changed')
    shutil.copytree(prior/'expert-assets',assets)
    shutil.copyfile(pair/'restore/weights.npz',assets/'viscosity-weights.npz')
    write(assets/'viscosity-initialization.json',dict(weights_sha256=sha(assets/'viscosity-weights.npz'),native_qualified_float32=True,
        managed_training_run_id=receipt['training_run_id'],managed_CPU_restore_run_id=receipt['restore_run_id'],
        pair_receipt_sha256=sha(pair/'receipt.json'),native_evaluation_sha256=sha(evaluation),
        parameters=243,coefficient_optimizer_updates=1000,no_analytic_coefficient_assignment=True))
    work={1:(20000,4000),2:(10000,2000),3:(6667,1334),4:(5000,1000)}
    summaries=[]
    for k,(updates,single)in work.items():
        config=read(prior/f'k{k}-config.json')
        config.update(schema='rheon-fixed-neural-viscosity-fit-v1',identifier=f'b{k}-fixed-viscosity',updates=updates,
            single_block_updates=single,schedule=[dict(update=(updates*4)//5+1,learning_rate=.0003)],
            viscosity_coefficients_frozen_from_start=True,
            initialization='Fresh shared cell, common seed; managed point/flux/pressure and managed243viscosity coefficients frozen',
            expert_asset_sha256={str(p.relative_to(assets)):sha(p)for p in sorted(assets.rglob('*'))if p.is_file()})
        write(output/f'k{k}-config.json',config)
        forwards=2*updates-single
        summaries.append(dict(horizons=k,optimizer_updates=updates,single_block_updates=single,
            block_forwards=forwards,physical_state_targets=forwards*k,encoder_calls=forwards,
            matched_FLOPs=False,sampling_seed=config['sampling_seed'],initialization_seed=config['seed']))
    write(output/'receipt.json',dict(status='prepared',source_sha256=sha(__file__),optimizer_updates=0,
        config_sha256={str(k):sha(output/f'k{k}-config.json')for k in work},work=summaries,
        calibration_pair_sha256=sha(pair/'receipt.json'),calibration_results_sha256=sha(DOCS/'viscosity-calibration-results.json'),
        original_inputs_sha256=sha(prior/'receipt.json'),expert_asset_sha256=config['expert_asset_sha256'],
        train_only=True,reserved_scenes_consumed=False,training_admitted=False))
    print('Prepared bounded K1/2/3/4 inputs,36000physical targets each; no optimizer updates',flush=True)


if __name__=='__main__':run()
