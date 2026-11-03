"""Expand the qualified original-training CUDA path to the22remaining fits."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require
from original_train_pressure_accuracy_02 import publication_gate
from original_train_pressure_cuda import verify_entry

HERE = Path(__file__).resolve().parent
PROTOCOL = RUN/'remaining-original-train-pressure-CUDA-protocol-01.json'


def run():
    publication_gate()
    output = RUN/'remaining-original-train-pressure-CUDA-qualification-01'
    require(not PROTOCOL.exists() and not output.exists(),'Fresh original-training batch required')
    pilot_path = RUN/'original-train-pressure-CUDA-pilot-02/fit-8992e19d389ea016/result.json'
    pilot = read(pilot_path)
    require(pilot['status'] == 'original_train_pressure_CUDA_accuracy_qualified' and pilot['cases'] == 128
            and not Path(f'/proc/{pilot["pid"]}').exists() and not pilot['unbound_lab_Python_modules']
            and pilot['same_weight_control_comparisons'][0]['all128_request_reference_and_output_hashes_exact'],
            'Qualified terminal original-train pilot and exact C66control required')
    predecessor = RUN/'original-train-pressure-CUDA-pilot-protocol-02.json'
    base = read(predecessor)
    material = read(ROOT/base['material']['path'])
    fit_ids = [r['fit_item_id'] for r in material['models'] if r['fit_item_id'] != pilot['fit_item_id']]
    require(len(fit_ids) == len(set(fit_ids)) == 22,'Exactly22unattempted fits required; do not repeat qualified pilot')
    sources = dict(base['source_sha256'])
    for path in HERE.glob('*.py'):
        spec = binding(path);sources[spec['path']] = spec['sha256']
    score_path = DOC/'reports/pressure-common-score-01.json'
    score = read(score_path)
    controls = {}
    for fit_id in fit_ids:
        verified = verify_entry(material,fit_id)
        entry = verified.entry
        for spec in [entry['original_receipt'],entry['original_configuration'],entry['source_witness'],*entry['sources'],*entry['outputs'].values()]:
            require(binding(ROOT/spec['path']) == {k:spec[k] for k in ('path','sha256')},'Original train batch material changed')
            sources[spec['path']] = spec['sha256']
        for name in entry['matching_measured_named_weight_candidates']:
            result = score['models'][name]['result']
            require(binding(ROOT/result['path']) == result,'Original named control result changed')
            prior = read(ROOT/result['path'])
            archive = (ROOT/result['path']).parent/prior['output_archive']['path']
            require(archive.is_file() and binding(archive)['sha256'] == prior['output_archive']['sha256'],'Original control archive unavailable')
            controls.setdefault(fit_id,[]).append(dict(identifier=name,result=result,archive=dict(binding(archive),bytes=archive.stat().st_size)))
            sources[result['path']] = result['sha256']
            spec = binding(archive);sources[spec['path']] = spec['sha256']
    value = dict(base,created_unix=time.time(),fit_ids=fit_ids,output_root=str(output.relative_to(ROOT)),source_sha256=sources,
        prior_controls=controls,accepted_pilot=binding(pilot_path),predecessor_protocol=binding(predecessor),
        worker='experiments/model-portfolio-migration-v1/original_train_pressure_accuracy_02.py',
        batch_owner='experiments/model-portfolio-migration-v1/run_original_train_pressure_accuracy.py',
        aggregate_deadline_seconds=3600,minimum_free_disk_bytes=512*1024**2,producer=binding(Path(__file__)))
    save(PROTOCOL,value)
    print('22remaining original-training fits preregistered; successful pilot is referenced, not repeated.',flush=True)


if __name__ == '__main__':
    run()
