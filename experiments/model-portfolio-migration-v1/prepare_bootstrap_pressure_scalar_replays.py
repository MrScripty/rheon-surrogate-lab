"""Admit exact original GN fits with explicit geometry and physics boundaries."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require


def run():
    output=RUN/'bootstrap-pressure-CUDA-protocol-01.json';require(not output.exists(),'Preserve first admission')
    material_path=RUN/'remaining-component-material-01.json';material=read(material_path)
    fits=['fit-0b2b6b6a8c42a47a','fit-4ca6c279be34e629']
    allowed={};dependencies=[binding(material_path)]
    physics=[]
    for fid in fits:
        entry=next(m for m in material['models'] if m['fit_item_id']==fid)
        for spec in [entry['original_receipt'],entry['original_configuration'],entry['source_witness'],
                *entry['sources'],*entry['outputs'].values(),*entry['inputs'].values()]:
            require(binding(ROOT/spec['path'])['sha256']==spec['sha256'],'Original bootstrap material changed')
            if spec['path'].endswith('.py'):allowed[spec['path']]=spec['sha256']
        splits=read(ROOT/entry['inputs']['splits']['path'])
        sizes={name:len(splits['shards'][name]['rows']) for name in ('train-0.npz','train-1.npz','validation.npz')}
        require(sizes=={'train-0.npz':144,'train-1.npz':144,'validation.npz':96},'Original12train/4validation trajectories times24frames required')
        physics.append(dict(fit_item_id=fid,split_manifest=entry['inputs']['splits'],native_frames=384,
            native_residual_build=splits['native_residual_origin']['native_build_receipt'],
            grid_counts_xyz=[16,16,16],fluid_density_kg_m3=1.25,spacing='Original independent hx/hy/hz,not one inferred scalar density'))
    for path in Path(__file__).parent.glob('*.py'):
        spec=binding(path);allowed[spec['path']]=spec['sha256']
    common=ROOT/'experiments/pressure-four-pass-distillation-v1/common.py'
    allowed[str(common.relative_to(ROOT))]=binding(common)['sha256']
    runtime=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    frozen_path=RUN/'discovery/frozen-pressure-requests-01.json';frozen=read(frozen_path)
    dependencies.extend([binding(runtime),binding(frozen_path),binding(ROOT/frozen['packet_manifest']['path'])])
    subset=[i for i,c in enumerate(frozen['cases']) if tuple(c['counts_xyz']) in ((16,16,16),(32,32,32))]
    require(len(subset)==2,'Original interface has exactlytwo applicable shared water cases')
    for name in ('bootstrap_pressure_cuda.py','bootstrap_pressure_scalar_replay.py','run_bootstrap_pressure_scalar_replays.py',
        'pressure_scalar_diagnostics.py','pressure_cases.py','python_source_coverage.py','cuda_forward_observer.py'):
        dependencies.append(binding(Path(__file__).with_name(name)))
    save(output,dict(status='two_original_bootstrap_GN_fits_CUDA_device_boundary_variant_admitted',created_unix=time.time(),
        producer=binding(Path(__file__)),dependencies=dependencies,material=binding(material_path),fit_ids=fits,
        original_physics=physics,admitted_lab_python=allowed,expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=8,
        worker_deadline_seconds=900,CPU_history_absolute_tolerance=1e-5,CPU_history_relative_tolerance=1e-4,
        runtime_manifest=str(runtime.relative_to(ROOT)),worker='experiments/model-portfolio-migration-v1/bootstrap_pressure_scalar_replay.py',
        output_root='runs/model-portfolio-migration-v1/bootstrap-pressure-CUDA-qualification-01',
        matched_native_case_count=768,own_native_cases=384,own_native_validation_cases=96,water_applicable_case_indices=subset,
        water_unsupported_count=126,counts_supported_by_source=[16,32],counts32not_yet_qualified=True,
        transformation='ChangeonlyCPU-to-CUDA device predicate anditsdiagnostic string in inheritedforward;entire operationAST and38selectedtensors exact. OriginalCPUfeatures,physicaloperator,andF64gauge/reconstruction retained.',
        exposure='Originaltwo cohorts andconsumed sharedwater requests;posthocdiagnostics,no independentholdout claim',
        primary_timing_samples=0,latency_qualified=False,prediction_arrays_saved=False,new_reference_generation=0,new_parameter_fits=0,
        metric_policy='Nativeunion andcommonwater subset remain separate physics/input populations. Pressurefield mean-free error,equation residual,andoriginalgauged normalized objective retain distinctnames.'))
    print('Admitted2fits:768matched native requests plus2water requests;126water shapes explicitlyunsupported.',flush=True)


if __name__=='__main__':run()
