"""Preregister one real CUDA path/control observation after native UI acceptance."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require
from original_train_pressure_accuracy import publication_gate
from original_train_pressure_cuda import verify_entry

HERE = Path(__file__).resolve().parent
PROTOCOL = RUN/'original-train-pressure-CUDA-pilot-protocol-01.json'
REPORT = DOC/'reports/original-train-pressure-CUDA-pilot-plan-01.json'
FIT_ID = 'fit-8992e19d389ea016'


def run():
    publication_gate()
    output = RUN/'original-train-pressure-CUDA-pilot-01'
    require(not PROTOCOL.exists() and not REPORT.exists() and not output.exists(),'Fresh preregistered pilot required')
    material_path = RUN/'remaining-pressure-material-01.json'
    material = read(material_path)
    verified = verify_entry(material,FIT_ID)
    score_path = DOC/'reports/pressure-common-score-01.json'
    score = read(score_path)
    result = score['models']['C66']['result']
    require(binding(ROOT/result['path']) == result,'Original C66control result changed')
    prior = read(ROOT/result['path'])
    require(prior['weights_sha256'] == verified.entry['outputs']['weights.npz']['sha256'],'Original C53-U/C66same-weight control required')
    archive = (ROOT/result['path']).parent/prior['output_archive']['path']
    require(archive.is_file() and binding(archive)['sha256'] == prior['output_archive']['sha256'],'Original prediction archive unavailable')
    sources = {str(p.relative_to(ROOT)):binding(p)['sha256'] for p in HERE.glob('*.py')}
    for spec in [verified.entry['original_receipt'],verified.entry['original_configuration'],verified.entry['source_witness'],
                 *verified.entry['sources'],*verified.entry['outputs'].values(),result,binding(archive),binding(material_path),binding(score_path)]:
        require(binding(ROOT/spec['path']) == {k:spec[k] for k in ('path','sha256')},'Original pilot material changed')
        sources[spec['path']] = spec['sha256']
    paths = [ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',
        ROOT/'experiments/pressure-four-pass-distillation-v1/common.py',
        ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/portable_model.py',
        DOC/'reports/native-matched-portfolio-UI-01.json',DOC/'reports/native-matched-generator-outcome-adoption-01.json']
    inputs = ROOT/'runs/water-early-pressure-v1/control/project/inputs'
    paths += [inputs/name for name in ['validation_splits.json','validation_dataset.npz','validation_dataset_extra.npz']]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = binding(path)['sha256']
    value = dict(schema='rheon.original-train-pressure-CUDA-qualification.v1',created_unix=time.time(),
        device='cuda',fit_ids=[FIT_ID],case_count=128,material=binding(material_path),source_sha256=sources,
        output_root=str(output.relative_to(ROOT)),host_cpu=4,expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',
        worker_deadline_seconds=1200,max_CUDA_allocated_bytes=2*1024**3,latency_claim=False,
        publication_acceptance=binding(DOC/'reports/native-matched-portfolio-UI-01.json'),
        worker=str((HERE/'original_train_pressure_accuracy.py').relative_to(ROOT)),
        prior_controls={FIT_ID:[dict(identifier='C66',result=result,archive=dict(binding(archive),bytes=archive.stat().st_size))]},
        acceptance='Original train mode retained; exact captured architecture/final tensors,all128native requests,labels excluded from prediction,all constructor/prediction forwards CUDA,common operator AST exact,source closure includes every actual imported labPythonmodule. Compare all128C66input/reference/output hashes; matching fields establish this population only and permit original archive reuse.',
        failure='Retain original adapter gap and stop for reconciliation; no architecture or policy fallback and no automatic retry.',
        new_parameter_updates=0,new_reference_generation=False,CPU_neural_fallback=False,full_portfolio_complete=False,
        producer=binding(Path(__file__)))
    save(PROTOCOL,value)
    save(REPORT,dict(status='one_original_train_CUDA_path_pilot_preregistered_after_actual_UI_acceptance',created_unix=time.time(),
        fit_item_id=FIT_ID,common_cases=128,prior_same_weight_control='C66',source_bindings=len(sources),protocol=binding(PROTOCOL),
        source_producer=binding(Path(__file__)),new_neural_forwards=0,latency_claim=False,full_portfolio_complete=False))
    print('Original C53-U training-output CUDA pilot preregistered:128cases/C66control; no model execution yet.',flush=True)


if __name__ == '__main__':
    run()
