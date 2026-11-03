"""Resolve authoritative field captures, including raw-report relative names.

Read-only inventory and scalar extraction admission. No deletion is authorized
by this report; scientific source assets and native payloads are excluded.
"""
from collections import defaultdict
from pathlib import Path
import re
import stat
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require


def run():
    report_path = DOC/'reports/prediction-capture-retention-inventory-02.json'
    admission_path = RUN/'pressure-statistics-admission-01.json'
    require(not report_path.exists() and not admission_path.exists(), 'Preserve original expanded inventory/admission')
    sources = {}
    candidates = {}

    def source(spec):
        path = ROOT/spec['path']
        actual = binding(path)
        require(actual['sha256'] == spec['sha256'], 'Bound scalar metadata changed')
        sources[spec['path']] = actual
        return read(path)

    def candidate(spec, owner, location):
        name = Path(spec['path'])
        # A basename belongs to its declaring raw result, never the repo root.
        path = (ROOT/owner).parent/name if len(name.parts) == 1 else ROOT/name
        require(not name.is_absolute() and '..' not in name.parts and path.is_relative_to(RUN)
            and path.resolve().is_relative_to(RUN.resolve()) and '/sources/' not in str(path)
            and path.suffix == '.npz' and 'weight' not in path.name and 'train' not in path.name,
            'Only task-owned explicitly declared derivative NPZ captures admitted')
        relative = str(path.relative_to(ROOT))
        item = candidates.setdefault(relative,dict(path=relative,sha256=spec['sha256'],references=[]))
        require(item['sha256'] == spec['sha256'], 'Conflicting capture identities')
        item['references'].append(dict(source=owner,location=location))
        return dict(path=relative,sha256=spec['sha256'])

    coverage_path = DOC/'reports/common-fit-benchmark-coverage-13.json'
    coverage = source(binding(coverage_path))
    common_path = DOC/'reports/pressure-common-score-01.json'
    common = source(binding(common_path))
    original_path = DOC/'reports/original-train-pressure-CUDA-accuracy-01.json'
    original = source(binding(original_path))
    entries = {m['fit_item_id']:m for m in original['rows']}
    fits = []
    for row in coverage['rows']:
        benchmark = row['default_benchmark']
        if benchmark is None or benchmark['group'] != 'pressure_request':
            continue
        name = benchmark['identifier']
        if name in common['models']:
            raw_spec = common['models'][name]['result']
            raw = source(raw_spec)
            archive = candidate(raw['output_archive'],raw_spec['path'],'/output_archive')
        else:
            entry = entries[row['fit_item_id']]
            raw_spec = entry['result']
            raw = source(raw_spec)
            archive = candidate(entry['prediction_archive'],str(original_path.relative_to(ROOT)),
                '/rows/'+row['fit_item_id']+'/prediction_archive')
            declared = candidate(raw['output_archive'],raw_spec['path'],'/output_archive')
            require(declared == archive, 'Original result and accepted prediction archive differ')
        require(len(raw['rows']) == 128, 'All original pressure cases required')
        fits.append(dict(fit_item_id=row['fit_item_id'],identifier=name,name=row['name'],
            accuracy_source=raw_spec,prediction_archive=archive,weights_sha256=raw['weights_sha256'],
            input_names=raw.get('input_names',['base','native_input_residual','cell_size_m','rho','dt']),
            input_declaration_source='original pressure qualification entry' if 'input_names' not in raw else 'captured raw result',
            derived_inputs=raw.get('derived_inputs'),execution_variant=raw.get('execution_variant',benchmark.get('execution_variant'))))
    prior_path = DOC/'reports/prediction-capture-retention-inventory-01.json'
    prior = source(binding(prior_path))
    for item in prior['candidates']:
        if (ROOT/item['path']).is_file():
            candidate(dict(path=item['path'],sha256=item['expected_sha256']),str(prior_path.relative_to(ROOT)),'/candidates')
    pattern = re.compile(r'(?:case\d{3,4}(?:_pressure|_middle|_end)?|pattern\d{3})\.npy')
    duplicates = defaultdict(list)
    for item in candidates.values():
        path = ROOT/item['path']
        observed = path.lstat()
        require(stat.S_ISREG(observed.st_mode) and binding(path)['sha256'] == item['sha256'],
            'Ordinary unchanged derivative capture required')
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(names and len(names) == len(set(names)) and all(pattern.fullmatch(n) for n in names),
                'Only audited forecast member names permitted')
        item.update(bytes=observed.st_size,inode=observed.st_ino,device=observed.st_dev,
            hard_links=observed.st_nlink,array_members=len(names),member_names_sha256=__import__('hashlib').sha256('\n'.join(names).encode()).hexdigest(),
            disposition='retained;expanded scalar and consumer audit incomplete;no deletion')
        duplicates[item['sha256']].append(item['path'])
    require(len(fits) == 150, 'Every accepted pressure fit must be admitted')
    frozen_path = DOC/'reports/frozen-pressure-requests-01.json'
    frozen_report = source(binding(frozen_path))
    contract = source(frozen_report['contract'])
    packet = source(contract['packet_manifest'])
    save(admission_path,dict(status='150pressure_fit_scalar_extraction_admitted_no_inference_or_deletion',
        created_unix=time.time(),producer=binding(Path(__file__)),source_reports=list(sources.values()),fits=fits,
        frozen_contract=frozen_report['contract'],packet_manifest=contract['packet_manifest'],
        packet_shards=[dict(path=str(((ROOT/contract['packet_manifest']['path']).parent/name).relative_to(ROOT)),
            sha256=spec['sha256'],bytes=spec['bytes']) for name,spec in zip(
                ('validation_dataset.npz','validation_dataset_extra.npz'),packet['shards'],strict=True)],
        expected_output_count=19200,prediction_deletion_allowed=False,new_model_forwards=0))
    save(report_path,dict(status='explicit_and_raw_relative_prediction_archives_verified_retained',created_unix=time.time(),
        producer=binding(Path(__file__)),source_reports=list(sources.values()),pressure_admission=binding(admission_path),
        candidates=list(candidates.values()),existing_files=len(candidates),existing_logical_bytes=sum(m['bytes'] for m in candidates.values()),
        equal_hash_multiple_paths=[dict(sha256=k,paths=v) for k,v in duplicates.items() if len(v)>1],
        scope='Previous32extant explicit report candidates plus authoritative127named/23original pressure result declarations. Not a recursive/unreferenced/native-payload inventory.',
        prediction_payload_values_read=0,model_payloads_opened=0,files_deleted=0,deletion_paused=True,
        new_model_forwards=0,native_mutations=0,full_portfolio_complete=False))
    print('Admitted150pressure fits;inventoried',len(candidates),'retained captures,',sum(m['bytes'] for m in candidates.values()),'bytes;no deletion.',flush=True)


if __name__ == '__main__':
    run()
