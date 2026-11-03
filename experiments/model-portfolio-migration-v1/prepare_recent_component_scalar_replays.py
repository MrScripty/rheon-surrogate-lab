"""Freeze the original eightfit populations and task-aware scalar semantics."""
from pathlib import Path
import hashlib
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require


def run():
    output=RUN/'recent-component-scalar-replay-protocol-01.json';require(not output.exists(),'Preserve first eightfit scalar admission')
    reports={};protocols={};allowed={};sources={}
    for kind,name in [('readout','macro-readout-field-GPU-accuracy-01.json'),('correction','internal-residual-GPU-accuracy-01.json'),('guided','guided-GPU-accuracy-01.json')]:
        path=DOC/'reports'/name;reports[kind]=binding(path);report=read(path)
        spec=report['protocol'];require(binding(ROOT/spec['path'])==spec,'Original protocol changed')
        protocols[kind]=spec
        for name,digest in read(ROOT/spec['path'])['source_sha256'].items():
            require(name not in sources or sources[name]==digest,'Conflicting original source authority');sources[name]=digest
            if name.endswith('.py'):allowed[name]=digest
    # Stream large archives;never materialize another corpus or project copy.
    for name,digest in sources.items():
        with (ROOT/name).open('rb') as stream:require(hashlib.file_digest(stream,'sha256').hexdigest()==digest,'Original source/data changed: '+name)
    for path in Path(__file__).parent.glob('*.py'):
        spec=binding(path);allowed[spec['path']]=spec['sha256']
    guide=read(ROOT/protocols['guided']['path']);ids=guide['fit_ids']
    jobs=[dict(name='readouts',kind='readout',host_cpu=4),dict(name='corrections',kind='correction',host_cpu=8),
        *[dict(name=fid,kind='guided',fit_item_id=fid,host_cpu=12+4*i) for i,fid in enumerate(ids)]]
    save(output,dict(status='eight_original_component_fits_scalar_replay_admitted',created_unix=time.time(),producer=binding(Path(__file__)),
        base_protocols=protocols,accepted_reports=reports,admitted_lab_python=allowed,jobs=jobs,
        runtime_manifest='runs/dependency-migration-v1/runtime/current-manifest-v2.json',
        output_root='runs/model-portfolio-migration-v1/recent-component-scalar-replay-01',
        worker='experiments/model-portfolio-migration-v1/recent_component_scalar_replay.py',
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',worker_deadline_seconds=7200,maximum_concurrent_workers=2,
        expected_fits=8,expected_output_records=17648,
        task_semantics=dict(readout='58physical1/60soutputs perfit;internal3/4stages notfutureleads',
            correction='783suppliedfine-history internalrequests perfit;zero-correctionbaseline notinitialstate',
            guided='2880blocks/7200physicaloutputs perfit at30/60/120outputs_per_s,K1–4'),
        metric_semantics='Acceptedtrain-flooredphysicalmetrics retainedexactly. Supplementary raw/numeric1e-30relative diagnostics remainseparate;include norms/floors/counts.',
        source_and_data_policy='Originalreferences/weights in place. Guidedtemporarysource views onlyPython plusverifiedlinks,removedonexit. No predictionfieldarchives.',
        latency_qualified=False,primary_timing_samples=0,new_parameter_fits=0,new_reference_generation=0,
        literal_scalar_tests_passed=7,scalar_test_source=binding(Path(__file__).with_name('test_full_state_scalar_diagnostics.py')),
        native_mutations=0,full_portfolio_complete=False))
    print('Admitted8fits/17648task-specificoutputrecords;fourjobs,twoatonce,no field archives.',flush=True)


if __name__=='__main__':run()
