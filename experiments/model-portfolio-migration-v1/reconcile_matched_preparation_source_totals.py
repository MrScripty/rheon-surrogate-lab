"""Correct advisory source totals after custody without changing admitted inputs."""
import base64
import copy
import hashlib
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,encode,binding,require


def run():
    folder = RUN/'matched-generator-GPU-measurements-02'
    destination = folder/'publication-summary.json'
    report_path = DOC/'reports/matched-preparation-source-byte-correction-01.json'
    require(not destination.exists() and not report_path.exists(),'Preserve original advisory metadata correction')
    custody_path = DOC/'reports/native-matched-generator-GPU-import-custody-01.json'
    custody = read(custody_path)
    owner_path = ROOT/custody['original_owner']['path']
    require(owner_path in {RUN/'native-matched-generator-GPU-import-01/status.json',
                          RUN/'native-matched-generator-GPU-import-reconciliation-01/status.json'}
            and binding(owner_path) == custody['original_owner'],'Matched custody must bind its accepted owner')
    owner = read(owner_path)
    require(owner['status'] == 'four_readable_matched_GPU_imports_native_metadata_and_maxima_passed'
            and owner['active_request'] is None and not Path(f'/proc/{owner["pid"]}').exists()
            and custody['status'] == 'four_readable_matched_GPU_tables_native_custody_exact'
            and custody['checkpoint'] == owner['last_acknowledged_checkpoint'],
            'Original matched import and after-custody must finish before correction')
    prep_path = folder/'preparation.json'
    previous = prep_path.read_bytes()
    preparation = read(prep_path)
    summary = copy.deepcopy(preparation)
    corrections = []
    for part in summary['packets']:
        path = folder/part['path']
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == part['sha256'] and len(raw) == part['bytes'],
                'Admitted packet changed')
        packet = read(path)
        for source in packet['sources']:
            body = base64.b64decode(source['data_base64'],validate=True)
            require(len(body) == source['bytes'] and hashlib.sha256(body).hexdigest() == source['sha256'],
                    'Retained source bytes/hash differ')
        actual = sum(source['bytes'] for source in packet['sources'])
        corrections.append(dict(K=part['K'],previous_advisory_source_bytes=part['retained_source_bytes'],actual_source_bytes=actual))
        part['retained_source_bytes'] = actual
    summary.update(status='published_matched_table_summary_with_actual_source_byte_totals',created_unix=time.time(),
        admitted_preparation=binding(prep_path),source_byte_correction_producer=binding(Path(__file__)),
        native_custody=binding(custody_path))
    destination.write_bytes(encode(summary))
    require(prep_path.read_bytes() == previous,'Original admitted preparation must remain byte-exact')
    report = dict(status='advisory_source_totals_corrected_original_admission_and_packets_unchanged',created_unix=time.time(),
        corrections=corrections,admitted_preparation=binding(prep_path),corrected_summary=binding(destination),
        native_custody=binding(custody_path),producer=binding(Path(__file__)),
        scientific_values_changed=False,packets_changed=False,native_mutations=0,new_neural_forwards=0,full_portfolio_complete=False)
    report_path.write_bytes(encode(report))
    print('All four advisory source totals corrected in publication-summary.json; original admission and packets unchanged.')


if __name__ == '__main__':
    run()
