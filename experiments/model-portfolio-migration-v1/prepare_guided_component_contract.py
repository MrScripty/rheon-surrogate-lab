"""Inspect original guided source archives and shared native data in place."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, PROJECT, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    output=RUN/'guided-component-contract-01.json'
    require(not output.exists(), 'Preserve original guided admission')
    publication,_=current_publication(ROOT)
    material_path=RUN/'remaining-component-material-01.json'
    selected=[r for r in read(material_path)['models'] if r['task'] in ('guided_shared_component','guided_pressure_component')]
    require(len(selected)==2, 'Both original guided fits required')
    observations={};models=[];data_spec=None;data_manifest=None
    def observe(spec):
        if spec['path'] not in observations:observations[spec['path']]=observed_file(ROOT,spec['path'],spec)
        require(observations[spec['path']]['sha256']==spec['sha256'], 'Conflicting original material identities')
        return observations[spec['path']]
    for entry in selected:
        for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],
                     *entry['inputs'].values(),*entry['outputs'].values()]:observe(spec)
        meta=read(ROOT/entry['inputs']['manifest']['path'])
        require(meta['model_source_sha256']==entry['inputs']['model_source']['sha256']
                and meta['normalization_sha256']==entry['inputs']['normalization']['sha256'],
                'Original source and loss-normalization bindings required')
        source=meta['source_manifest'];members=source['members'];keepers={};python_bytes=0
        with zipfile.ZipFile(ROOT/entry['inputs']['model_source']['path']) as archive:
            require(len(archive.infolist())==len(members) and set(archive.namelist())==set(members)
                    and sum(i.file_size for i in archive.infolist())<=128*1024**2, 'Finite exact original source archive required')
            for name,expected in members.items():
                path=Path(name);info=archive.getinfo(name)
                require(not path.is_absolute() and '..' not in path.parts
                        and not stat.S_ISLNK(info.external_attr>>16), 'Ordinary safe source member required')
                raw=archive.read(name)
                require(hashlib.sha256(raw).hexdigest()==expected, 'Original captured source member changed')
                if path.suffix=='.py':
                    python_bytes+=len(raw)
                    continue
                keeper=ROOT/name
                if not keeper.is_file() or keeper.is_symlink() or hashlib.sha256(keeper.read_bytes()).hexdigest()!=expected:
                    keeper=PROJECT/'artifacts/sha256'/expected
                require(keeper.is_file() and not keeper.is_symlink()
                        and hashlib.sha256(keeper.read_bytes()).hexdigest()==expected,
                        'An in-place exact keeper is required for non-Python payload: '+name)
                keepers[name]=observe(dict(path=str(keeper.relative_to(ROOT)),sha256=expected,bytes=len(raw)))
        current=observe(entry['inputs']['part_00'])
        if data_spec is None:
            data_spec=current
            with zipfile.ZipFile(ROOT/current['path']) as archive:
                require(archive.getinfo('manifest.json').file_size<=32*1024**2, 'Bounded original native data manifest required')
                data_manifest=json.loads(archive.read('manifest.json'))
        require(current==data_spec and meta['archive']=={k:current[k] for k in ('bytes','sha256')}
                and hashlib.sha256(json.dumps(data_manifest,sort_keys=True).encode()).hexdigest()==meta['bank_semantic_sha256'],
                'Both original fits must share this exact native data corpus and semantics')
        config=read(ROOT/entry['original_configuration']['path'])
        probe_path=(ROOT/entry['outputs']['weights.npz']['path']).parent/'probe.npy'
        receipt=read(ROOT/entry['original_receipt']['path'])
        probe=observe(dict(path=str(probe_path.relative_to(ROOT)),**receipt['outputs']['probe.npy']))
        models.append(dict(fit_item_id=entry['fit_item_id'],name=entry['name'],task=entry['task'],
            model_source=entry['inputs']['model_source'],manifest=entry['inputs']['manifest'],
            normalization=entry['inputs']['normalization'],weights=entry['outputs']['weights.npz'],
            inventory=entry['selected_inventory'],history=entry['outputs']['history.json'],
            original_receipt=entry['original_receipt'],original_configuration=entry['original_configuration'],
            probe=probe,probe_identifier=config['probe_identifier'],probe_start_fine_step=config['probe_start_fine_step'],
            captured_model_family=source['model_family'],configuration=source['configuration'],
            actual_fit_core_module='guided_shared_fit_core_v3' if entry['task']=='guided_shared_component' else 'guided_pressure_fit_core_v1',
            actual_model_module='guided_shared_residual_v3' if entry['task']=='guided_shared_component' else 'guided_pressure_residual_v1',
            non_Python_keepers=keepers,temporary_Python_source_bytes=python_bytes,
            original_native_source_members=len(members),captured_source_schema=source['schema']))
    masters=data_manifest['masters'];require(len(masters)==24 and all(v['split']=='train' for v in masters.values()), 'Original24train masters required')
    cases=[]
    for identifier,master in masters.items():
        starts=sorted({w['start_fine_step'] for w in master['windows']})
        expected_starts=sorted({w['start_fine_step'] for w in meta['qualified_index']['masters'][identifier]['windows']})
        require(starts==expected_starts and len(starts) in (9,11)
                and len(master['windows'])==12*len(starts), 'All original base and event-targeted starts/all12configurations required')
        for window in master['windows']:
            rate,k=window['output_steps_per_s'],window['K']
            require(rate in (30,60,120) and k in (1,2,3,4)
                    and window['target_fine_steps']==[window['start_fine_step']+lead*(240//rate) for lead in range(1,k+1)],
                    'Whole physical output intervals required')
            cases.append(dict(identifier=identifier,scene=master['scene'],**window))
    require(len(cases)==2880 and sum(c['K'] for c in cases)==7200, 'Full2880blocks/7200physical outputs perfit required')
    save(output,dict(status='two_original_guided_fits_captured_sources_and_full24_data_bound_not_benchmarked',
        created_unix=time.time(),producer=binding(Path(__file__)),publication=binding(publication),
        material=binding(material_path),models=models,data_archive=data_spec,
        data_manifest_semantic_sha256=hashlib.sha256(json.dumps(data_manifest,sort_keys=True).encode()).hexdigest(),
        original_material_observations=list(observations.values()),cases=cases,training_parents=24,
        starts_per_parent_histogram=dict(sorted(Counter(str(len({w['start_fine_step'] for w in m['windows']})) for m in masters.values()).items())),
        selected_current_windows=240,physical_configurations_per_start=12,blocks_per_fit=2880,physical_outputs_per_fit=7200,
        by_cells_per_metre=dict(sorted(Counter(str(c['scene']['cells_per_metre']) for c in cases).items())),
        by_counts_xyz=dict(sorted(Counter('x'.join(map(str,c['scene']['counts'])) for c in cases).items())),
        source_policy='Temporary regular captured Python source files preserve original __file__/ROOT behavior. Every non-Python member is a verified symlink to an existing in-place keeper. No weights,dataset,project or compiled payload copied. Remove only the owned temporary source view after closing the model.',
        input_policy='Current native PVT plus prescribed controls/events;future state/RHS targets remain scoring-only. Direct encode/endpoint/interior calls require explicit CUDA tensor guards as well as nn.Module hooks.',
        output_policy='Original complete guided composition emits every requested physical output. K advances K/output_rate seconds;not internal subdivisions of one output interval. Report component-only and composition timings separately if measured.',
        inference_device='cuda',CPU_neural_forwards=0,accuracy_qualified=False,latency_qualified=False,
        independent_holdout=False,own_history_drift=False,new_parameter_fits=0,new_reference_generation=0,
        model_dataset_or_project_copies=0,native_mutations=0,full_portfolio_complete=False))
    print('Both guided fits admitted:24parents,2880blocks/7200outputs each;shared2.81GBdataset verified in place.',flush=True)


if __name__=='__main__':run()
