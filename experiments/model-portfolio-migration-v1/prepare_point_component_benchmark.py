"""Preregister original80pattern publishedF32point component evaluation."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require
from component_material import current_publication,observed_file

HERE = Path(__file__).resolve().parent
PROTOCOL = RUN/'point-component-GPU-protocol-01.json'


def run():
    publication,_ = current_publication(ROOT)
    output = RUN/'point-component-GPU-benchmark-01'
    require(not PROTOCOL.exists() and not output.exists(),'Fresh original point protocol required')
    material_path = RUN/'remaining-component-material-01.json'
    material = read(material_path)
    fid = 'fit-d106ea978eb8ed52'
    entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
    receipt = read(ROOT/entry['original_receipt']['path'])
    probe_path = Path(entry['original_receipt']['path']).parent/'probe.npy'
    probe = observed_file(ROOT,probe_path,receipt['outputs']['probe.npy'])
    sources = {}
    for path in HERE.glob('*.py'):
        spec = binding(path);sources[spec['path']] = spec['sha256']
    for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),*entry['inputs'].values(),probe]:
        observed_file(ROOT,spec['path'],spec);sources[spec['path']] = spec['sha256']
    # Preserve the already-qualified virtual Torch namespace implementations.
    previous = read(RUN/'original-pressure-GPU-timing-pilot-protocol-01.json')
    for path,sha in previous['source_sha256'].items():
        if Path(path).is_absolute():
            sources[path] = sha
    runtime_path = ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    save(PROTOCOL,dict(schema='rheon.point-component-GPU-benchmark.v1',created_unix=time.time(),fit_item_id=fid,
        material=binding(material_path),publication_acceptance=binding(publication),runtime_manifest=binding(runtime_path),
        source_sha256=sources,output_root=str(output.relative_to(ROOT)),original_probe={k:probe[k] for k in ('path','sha256')},
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,cases=80,train_cases=64,consumed_development_cases=16,
        worker='experiments/model-portfolio-migration-v1/point_component_benchmark.py',
        warmups_per_case=5,samples_per_case=10,calibration_pairs_per_case=4,worker_deadline_seconds=600,aggregate_deadline_seconds=1500,
        maximum_CUDA_allocated_bytes=2*1024**3,maximum_RSS_bytes=8*1024**3,minimum_free_disk_bytes=512*1024**2,
        phase_order=[0,1],case_order='0..79then79..0in distinct processes',published_parameter_dtype='float32',prediction_dtype='float32',
        reference_dtype='float64',autocast=False,tf32=False,deterministic=True,
        timing_scope='OwnedCPUF32velocity through finite/type/shape validation,packing,H2D,unchangedcapturedCUDAPointSamplingNet,independentownedCPUoutput,D2H/sync. Hashing,score,observer,setup/first/warmup anddatafileload excluded fromprimary.',
        timing_units='ms/local_point_request',physical_simulation_steps=0,native_counts_xyz=[12,12,12],padded_tensor_spatial_shape=[13,13,13],
        destination_channel_order=['vx_face','vy_face','vz_face','tracer_cell'],
        metric_scope='Joint12channel andeach3channel destination-group masked negative sampled-velocity errors,not pressure orfull-state error. Report targetnorm/floor;nonzero-target relative maxima andzero-target absolute errors distinct.64fitting/16consumeddevelopment,no independentholdout. Time is for all12channels;not divided bydestination count.',
        CPU_probe_policy='Read the original2fieldpublishedCPUprobe only. No newCPUforward. CompareGPUF32toCPUF32with2gamma81*max|input|*maxoutputchannelweightL1 +float32tiny,gamma81=81u/(1-81u),u=float32eps/2. This bounds two81termFP32contractions;not bitwiseCPU/GPUequivalence.',
        calibration_policy='4pairedon/offcalls percase;alternate orderbyphase,index,pair. Observer removed forprimary/off;overhead notsubtracted ornegative-clipped.',
        resource_scope='Before/every20cases/after computePID/GPUclock/temperature/load snapshots;not continuous exclusivity.',
        new_parameter_updates=0,new_reference_generation=False,CPU_neural_forward_allowed=False,model_or_dataset_copies=0,
        producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('Original point972F32coefficients/80patterns/twoCUDAphases preregistered;no inference executed.',flush=True)


if __name__ == '__main__':
    run()
