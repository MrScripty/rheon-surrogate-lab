"""Bind the qualified managed sampler and frozen learned pressure assets."""
import shutil
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=BASE/'temporal-displacement-inputs-01'
    if output.exists():raise ValueError('Fresh displacement inputs required')
    pair=BASE/'point-sampling-pair-01';receipt=read(pair/'receipt.json')
    evaluation_path=BASE/'point-sampling-native-evaluation-01.json';evaluation=read(evaluation_path)
    if receipt['status']!='completed'or not all(receipt['artifact_parity'].values())or evaluation['status']!='passed_component_qualification':
        raise ValueError('Managed exact-restore native-qualified point coefficients required')
    for name,digest in evaluation['source_evidence'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified point evidence changed')
    config=read(BASE/'b2-coupled-project-01/config.json')
    old_assets=BASE/'temporal-coupled-assets-01'
    for name,digest in config['expert_asset_sha256'].items():
        if sha(old_assets/name)!=digest:raise ValueError('Qualified original neural pressure/flux asset changed')
    output.mkdir(parents=True);assets=output/'expert-assets';shutil.copytree(old_assets,assets)
    shutil.copyfile(pair/'restore/weights.npz',assets/'point-weights.npz')
    write(assets/'point-initialization.json',dict(weights_sha256=sha(assets/'point-weights.npz'),native_qualified_float32=True,
        managed_training_run_id=receipt['training_run_id'],managed_CPU_restore_run_id=receipt['restore_run_id'],
        pair_receipt_sha256=sha(pair/'receipt.json'),native_evaluation_sha256=sha(evaluation_path),
        parameters=972,coefficient_optimizer_updates=1000,no_analytic_coefficient_assignment=True))
    remove=('full_fit_registration_sha256','continuation_weights_sha256','continuation_inventory','auxiliary_provisional_weight')
    for name in remove:config.pop(name,None)
    config.update(schema='rheon-temporal-displacement-fit-v1',architecture='displacement',
        updates=10000,single_block_updates=2000,learning_rate=.001,
        schedule=[dict(update=8001,learning_rate=.0003)],seed=2026102045,displacement_auxiliary_weight=1.,
        initialization='Fresh shared neural fluid cell, qualified pretrained pressure/flux and managed 972-coefficient point calibration',
        inference='One shared encoder, fused learned point/local projections, neural displacements, fixed convex field readouts, learned pressure and momentum; all K states',
        decoder_representation='Neural point displacement plus learned second-order curvature; shared velocity/tracer transport readout',
        loss='Original step-unit full-state objective plus native-qualified displacement labels in alpha-squared units; labels only after prediction',
        expert_asset_sha256={str(p.relative_to(assets)):sha(p)for p in sorted(assets.rglob('*'))if p.is_file()})
    for horizons in (1,2,3,4):
        write(output/f'k{horizons}-config.json',dict(config,horizons=horizons,identifier=f'b{horizons}-displacement'))
    write(output/'receipt.json',dict(status='prepared',source_sha256=sha(__file__),optimizer_updates=0,
        qualified_point_pair_sha256=sha(pair/'receipt.json'),qualified_point_evaluation_sha256=sha(evaluation_path),
        config_sha256={str(k):sha(output/f'k{k}-config.json')for k in (1,2,3,4)},expert_asset_sha256=config['expert_asset_sha256'],
        normalization_sha256=config['original_normalization_sha256'],step_scales_sha256=config['step_scale_statistics_sha256'],
        no_new_pressure_coefficients=True,reserved_scenes_consumed=False))
    print('Prepared learned displacement inputs for K1/2/3/4; no full-state optimizer updates',flush=True)


if __name__=='__main__':run()
