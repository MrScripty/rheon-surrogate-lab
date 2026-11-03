"""Freeze the original48parameter/77998point recurrent-head CUDA benchmark."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    publication,_ = current_publication(ROOT)
    path = RUN/'sampler-component-GPU-protocol-01.json';output = RUN/'sampler-component-GPU-benchmark-01'
    require(not path.exists() and not output.exists(), 'Fresh original sampler protocol required')
    material_path = RUN/'remaining-component-material-01.json';material = read(material_path)
    fid = 'fit-11a2f63a94a9168e';entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
    require(entry['serialized_tensor_elements'] == 48 and entry['parameter_dtypes'] == ['<f8'], 'Original48F64parameters required')
    sources = {}
    for source in Path(__file__).parent.glob('*.py'):
        spec = binding(source);sources[spec['path']] = spec['sha256']
    for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),
                 *entry['inputs'].values(),binding(material_path)]:
        observed_file(ROOT,spec['path'],spec);sources[spec['path']] = spec['sha256']
    manifest = read(ROOT/entry['inputs']['manifest']['path'])
    for name,sha in manifest['source_evidence'].items():
        if name.endswith('/receipt.json'):
            observed_file(ROOT,name,dict(sha256=sha));sources[name] = sha
    teacher = ROOT/'experiments/full-ai-tight-targets-v1/prepare_sampler_training.py';spec = binding(teacher)
    require(manifest['source_sha256'][spec['path']] == spec['sha256'], 'Original sampler label preparation source differs')
    sources[spec['path']] = spec['sha256']
    prior = read(RUN/'shared-component-GPU-protocol-02.json')
    for name,sha in prior['source_sha256'].items():
        if Path(name).is_absolute():
            sources[name] = sha
    runtime = ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    save(path,dict(schema='rheon.sampler-component-GPU-benchmark.v1',created_unix=time.time(),
        fit_item_id=fid,material=binding(material_path),publication_acceptance=binding(publication),runtime_manifest=binding(runtime),
        source_sha256=sources,output_root=str(output.relative_to(ROOT)),worker='experiments/model-portfolio-migration-v1/sampler_component_benchmark.py',
        phase_order=[0,1],cases=576,points=77998,expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,
        warmups_per_case=5,samples_per_case=10,calibration_pairs_per_case=4,worker_deadline_seconds=900,aggregate_deadline_seconds=2100,
        minimum_free_disk_bytes=512*1024**2,maximum_CUDA_allocated_bytes=2*1024**3,maximum_RSS_bytes=8*1024**3,
        timing_units='ms/local_sampler_group',physical_simulation_steps=0,
        timing_scope='OwnedCPU pointF64/patchF32/kindI64/alphaF64 validation/packing/H2D,unchangedoriginalCUDA LearnedSamplerHead,allmiddle/endownedCPUF64vectors,D2H/sync. Per130-136point originalgroup;not a fullgrid orsimulationstep. Originaldtype conversions insideforward retained.',
        output_units='middle:cell-coordinate midpoint shift/alpha,velocity-equivalentm/s;end:negative sampledvelocitym/s. Not finalpositions orcomplete advected state.',
        accuracy_scope='Each middle/endstage:grouprelativeL2,absoluteRMS/max,targetnorm/floor,nonzero-target perpointrelative maximum,zero-targetpoint absoluteL2. All576groups/77998points;carrier,density,grid andjoint groups distinct. Scales not applied to physical-error metrics.',
        population='16originalfitting scenes/fourgridcontexts/cpm8and16/states0,16,...128/fourcarriers. Originaltargets include boundary-clamped midpoint labels. Noindependentholdout ortrajectory drift.',
        alpha_policy='Explicitoriginalalpha=cells_per_metre/60 matchesdt/h atdt1/60. Joinoriginalreceipts forphysicalmetadata;no inference fromnames.',
        prediction_dtype='float64',autocast=False,tf32=False,deterministic=True,primary_observer_removed=True,
        observer_overhead_subtracted=False,resource_scope='Before/every144groups/after snapshots;not continuous exclusivity',
        CPU_neural_forward_allowed=False,new_parameter_fits=0,new_reference_generation=False,weight_or_dataset_copies=0,
        producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('Original sampler48F64parameters/576groups/77998points/twoCUDAphases frozen;no inference.',flush=True)


if __name__ == '__main__':
    run()
