"""Task-specific scalar extraction for ten component fits; no model execution.

Modal/shared pressure, future q, local viscosity, point sampling and local
middle/end sampler outputs retain their distinct meanings and populations.
Only the original q data decoder uses CPU Torch, guarded against every neural
forward. No learned solver/filter, constructor or weight payload is loaded.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from pressure_scalar_diagnostics import field_summary, error_summary, equation


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def run():
    import numpy as np
    output = RUN/'other-component-field-statistics-02.jsonl.gz'
    report_path = DOC/'reports/other-component-field-statistics-02.json'
    require(not output.exists() and not report_path.exists(), 'Preserve first component scalar extraction')
    os.sched_setaffinity(0,{4 if 4 in os.sched_getaffinity(0) else min(os.sched_getaffinity(0))})
    began = time.monotonic()
    sources = {}
    contexts = {}
    descriptions = {}
    counts = Counter()
    checks = Counter()
    captures = {}
    guard_evidence = None

    def load(spec):
        if isinstance(spec,Path):
            spec = binding(spec)
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Source changed: '+spec['path'])
        sources[spec['path']] = dict(path=spec['path'],sha256=spec['sha256'])
        return read(ROOT/spec['path'])

    def capture(spec):
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Accepted output archive changed')
        captures[spec['path']] = dict(path=spec['path'],sha256=spec['sha256'])
        return np.load(ROOT/spec['path'],allow_pickle=False)

    def compare(actual,expected,label):
        require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual-expected) <= 1e-14+1e-11*abs(expected), 'Accepted scalar differs: '+label)
        checks[label] += 1

    def paired(prediction,target):
        p,r = np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
        require(p.shape == r.shape and p.size, 'Matched nonempty output/target layout required')
        return dict(prediction=field_summary(p),reference=field_summary(r),error=error_summary(p-r,r))

    pending = output.with_suffix('.pending')
    with pending.open('xb') as destination:
        with gzip.GzipFile(filename='',mode='wb',fileobj=destination,mtime=0,compresslevel=6) as compressed:
            def emit(fid,index,statistics,stage=None):
                compressed.write((json.dumps(dict(fit_item_id=fid,case_index=index,stage=stage,statistics=statistics),
                    separators=(',',':'),sort_keys=True,allow_nan=False)+'\n').encode())
                counts[fid] += 1

            # These native pressure requests share one exact144case reference
            # bank. They do not share the CNN128 input/timing contract.
            from modal_component_data import load_fields
            modal = load(DOC/'reports/modal-component-GPU-benchmark-01.json')
            protocol = load(modal['protocol'])
            contract = load(modal['contract'])
            fields = load_fields(contract,protocol)
            scene_metadata = {}
            for item in contract['rows']:
                path = (ROOT/item['source']).parent/'receipt.json'
                name = str(path.relative_to(ROOT))
                if name not in scene_metadata:
                    receipt = load(dict(path=name,sha256=protocol['source_sha256'][name]))
                    scene_metadata[name] = receipt['scene']
            contexts['native_pressure144'] = dict(cases=contract['rows'],scenes_by_receipt=scene_metadata,
                input='Native F64 RHS, h and fluid mass density; no complete state input',
                output='Cell-zero-gauged F64 pressure; no physical output interval inferred from request index')
            shared = load(DOC/'reports/shared-component-GPU-benchmark-02.json')
            prepared = next(m for m in shared['modes'] if m['mode'] == 'prepared')
            other = next(m for m in shared['modes'] if m['mode'] != 'prepared')
            require([r['output_sha256'] for r in prepared['rows']] == [r['output_sha256'] for r in other['rows']],
                'Shared cache modes must have equal accepted outputs before reusing scalar extraction')
            for model in [*modal['models'],prepared]:
                fid = model['fit_item_id']
                descriptions[fid] = dict(population='native_pressure144',task='native_RHS_pressure_component',
                    input='Owned native RHS, h/rho; declared modal/shared fixed geometry',
                    output='Pressure only',baseline='No supplied pressure base in this request interface',
                    extra_execution_modes_same_outputs=['prepared','unprepared'] if model is prepared else [])
                with capture(model['prediction_archive']) as archive:
                    require(set(archive.files) == {f'case{i:03d}' for i in range(144)}, 'Complete modal pressure outputs required')
                    for index,(case,row) in enumerate(zip(fields,model['rows'],strict=True)):
                        p = archive[f'case{index:03d}']
                        r,q = case['reference'],case['rhs']
                        require(digest(p) == row['output_sha256'] and digest(r) == row['reference_pressure_sha256']
                            and digest(q) == row['input_rhs_sha256'] and row['identity'] == case['identity'], 'Modal request/output identity differs')
                        difference = p-r
                        centered = difference-difference.mean()
                        center_target = r-r.mean()
                        residual = equation(p,[case['h']]*3,case['rho'])-q
                        metrics = dict(prediction_pressure=field_summary(p),reference_pressure=field_summary(r),
                            input_rhs=field_summary(q),pressure_mean_free_error=error_summary(centered,center_target),
                            pressure_gauged_error=error_summary(difference,r),equation_residual=error_summary(residual,q),
                            pressure_gauge_offset=float(p.mean()-r.mean()))
                        for name,value in [('pressure_absolute_rms',metrics['pressure_mean_free_error']['rms']),
                            ('pressure_absolute_max',metrics['pressure_mean_free_error']['maximum_absolute']),
                            ('pressure_mean_free_relative_l2',metrics['pressure_mean_free_error']['relative_l2']),
                            ('common_equation_relative_l2',metrics['equation_residual']['relative_l2'])]:
                            compare(value,row['metrics'][name],name)
                        emit(fid,index,metrics)

            # Two different local linear operator teachers. Their masks and
            # target units are part of their contracts, not full-state metrics.
            material = None
            for kind in ('point','viscosity'):
                report = load(DOC/f'reports/{kind}-component-GPU-benchmark-01.json')
                material = load(report['material'])
                fid = report['fit_item_id']
                entry = next(m for m in material['models'] if m['fit_item_id'] == fid)
                teacher_spec = entry['inputs']['teacher']
                teacher_manifest = load(entry['inputs']['teacher_manifest'])
                require(binding(ROOT/teacher_spec['path'])['sha256'] == teacher_spec['sha256'], 'Original teacher changed')
                with np.load(ROOT/teacher_spec['path'],allow_pickle=False) as teacher:
                    bank = {n:teacher[n] for n in teacher.files}
                require(set(bank) == {'velocity','target','mask'} and all(digest(v) == teacher_manifest['inventory'][n]['sha256'] for n,v in bank.items()),
                    'Original local input/reference arrays differ')
                mask = bank['mask'][0]
                require(np.isin(mask,[0,1]).all() and np.count_nonzero(mask), 'Nonempty binary physical mask required')
                active = mask.astype(bool)
                phase_rows = []
                for spec in report['raw_results']:
                    phase = load(spec)
                    phase_rows.append({r['pattern_index']:r for r in phase['rows']})
                require(len(phase_rows) == 2 and all(len(rows) == 80 for rows in phase_rows),
                    'Both raw local worker populations required for target/mask identity')
                contexts[kind+'80'] = dict(counts_xyz=teacher_manifest['counts_xyz'],
                    target=teacher_manifest['target'],train_indices=teacher_manifest['train_indices'],
                    development_indices=teacher_manifest['development_indices'],
                    grid_density='not_applicable_to_synthetic_local_patterns;do_not_infer_from_native_qualification_geometries')
                descriptions[fid] = dict(population=kind+'80',task=kind+'_local_operator',
                    input='Synthetic local XYZ velocity pattern',output=teacher_manifest['target'],
                    baseline='No full physical advance or generic unchanged-input baseline claimed')
                with capture(report['prediction_archive']) as archive:
                    require(set(archive.files) == {f'pattern{i:03d}' for i in range(80)}, 'Complete synthetic output patterns required')
                    for index,row in enumerate(report['rows']):
                        p,r,v = archive[f'pattern{index:03d}'],bank['target'][index],bank['velocity'][index]
                        require(digest(p) == row['output_sha256'] and digest(v) == row['input_sha256'], 'Local input/output identities differ')
                        for phase in phase_rows:
                            original = phase[index]
                            require(digest(r) == original['target_sha256'] and digest(mask) == original['mask_sha256']
                                and digest(p) == original['output_sha256'] and digest(v) == original['input_sha256'],
                                'Local raw target/mask/input/output identities differ')
                        values = paired(p[active],r[active])
                        values['input_velocity'] = field_summary(v)
                        for name,key in [('masked_relative_l2','relative_l2'),('masked_absolute_rms','rms'),('masked_absolute_max','maximum_absolute')]:
                            compare(values['error'][key],row['metrics'][name],name)
                        if kind == 'point':
                            destinations = {}
                            for group,name in enumerate(report['destination_channel_order']):
                                sl = slice(3*group,3*group+3)
                                selected = active[sl]
                                destinations[name] = paired(p[sl][selected],r[sl][selected])
                                for metric,key in [('masked_relative_l2','relative_l2'),('masked_absolute_rms','rms')]:
                                    compare(destinations[name]['error'][key],row['destination_metrics'][name][metric],'destination.'+metric)
                            values['destination_sampled_vectors'] = destinations
                        emit(fid,index,values)

            # Middle/end are two local velocity-equivalent vector targets,
            # not two complete future simulation frames.
            from sampler_component_data import load_groups
            sampler = load(DOC/'reports/sampler-component-GPU-benchmark-01.json')
            sampler_protocol = load(sampler['protocol'])
            sampler_material = load(sampler['material'])
            fid = sampler['fit_item_id']
            entry = next(m for m in sampler_material['models'] if m['fit_item_id'] == fid)
            for spec in entry['inputs'].values():
                require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Sampler input changed')
            groups = load_groups(entry)
            contexts['sampler576'] = dict(cases=[g['identity'] for g in groups],output_units=sampler_protocol['output_units'])
            descriptions[fid] = dict(population='sampler576',task='local_sampler_group',
                input='Original point, local velocity patch, carrier kind, alpha=h_inverse*dt',
                output=sampler_protocol['output_units'],physical_output_leads='not_applicable;middle/end are internal local targets')
            with capture(sampler['prediction_archive']) as archive:
                require(set(archive.files) == {f'case{i:03d}_{stage}' for i in range(576) for stage in ('middle','end')}, 'Every local target required')
                for index,(group,row) in enumerate(zip(groups,sampler['rows'],strict=True)):
                    require(group['identity'] == row['identity'], 'Sampler group differs')
                    for name in ('point','patch','kind','alpha'):
                        require(digest(group['arrays'][name]) == row['input_sha256'][name], 'Sampler input differs')
                    for stage in ('middle','end'):
                        p,r = archive[f'case{index:03d}_{stage}'],group['arrays'][stage]
                        require(digest(p) == row['output_sha256'][stage] and digest(r) == row['target_sha256'][stage], 'Sampler output/target differs')
                        values = paired(p,r)
                        values['XYZ_vector_components'] = {axis:paired(p[:,j],r[:,j]) for j,axis in enumerate(('x','y','z'))}
                        values['input_point'] = field_summary(group['arrays']['point'])
                        values['input_velocity_patch'] = field_summary(group['arrays']['patch'])
                        for metric,key in [('relative_l2','relative_l2'),('absolute_rms','rms'),('absolute_max','maximum_absolute')]:
                            compare(values['error'][key],row['metrics'][stage][metric],'sampler.'+metric)
                        emit(fid,index,values,stage)

            # The captured q data API computes only fixed features/controls.
            # A global guard forbids any CPU neural forward before it runs.
            q_report = load(DOC/'reports/causal-request-component-GPU-benchmark-01.json')
            q_material = load(q_report['material'])
            entries = {m['fit_item_id']:m for m in q_material['models']}
            import torch
            from cuda_forward_observer import CUDAForwardObserver
            from causal_request_component_data import load_original, owned_case, HELPERS
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            require(not torch.cuda.is_initialized(), 'Scalar extraction must not initialize CUDA')
            guard = CUDAForwardObserver(torch)
            guard.enable('fixed_q_decoder_only_no_neural_execution')
            try:
                first_entry = entries[q_report['models'][0]['fit_item_id']]
                for spec in first_entry['inputs'].values():
                    require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Captured q data input changed')
                module,_,norm,parents,caches = load_original(first_entry)
                for name in HELPERS:
                    path = Path(sys.modules[name].__file__).resolve()
                    spec = next(s for s in first_entry['sources'] if s['original_name'] == name+'.py')
                    require(path == ROOT/spec['path'] and binding(path)['sha256'] == spec['sha256'], 'Captured fixed decoder source ownership differs')
                    sources[spec['path']] = dict(path=spec['path'],sha256=spec['sha256'])
                contexts['future_q2048'] = dict(scenes={p['identifier']:p['scene'] for p in parents},
                    normalized_q_to_physical_scale=norm['scales']['q'],
                    cases=[row['identity'] for row in q_report['models'][0]['rows']])
                for model in q_report['models']:
                    fid = model['fit_item_id']
                    descriptions[fid] = dict(population='future_q2048',task='normalized_future_RHS_request',
                        input='Current normalized velocity, known forcing, h/dt/rho/mu; pressure/q labels excluded',
                        output='Normalized future q only; no pressure solve or state transport',
                        baseline='No extra q input or full-state persistence baseline assumed')
                    with capture(model['prediction_archive']) as archive:
                        require(set(archive.files) == {f'case{i:04d}' for i in range(2048)}, 'All future-q outputs required')
                        for index,row in enumerate(model['rows']):
                            identity,host,target = owned_case(module,parents,caches,index)
                            require(all(row['identity'][k] == v for k,v in identity.items())
                                and {k:digest(v) for k,v in host.items()} == row['input_sha256']
                                and digest(target) == row['target_sha256'], 'Future-q input/target identity differs')
                            p = archive[f'case{index:04d}']
                            require(digest(p) == row['output_sha256'], 'Future-q output identity differs')
                            values = paired(p,target)
                            values['inputs'] = {k:field_summary(v) for k,v in host.items()}
                            values['physical_q_absolute_rms'] = values['error']['rms']*norm['scales']['q']
                            for metric,key in [('q_relative_l2','relative_l2'),('q_absolute_rms_normalized','rms'),('q_absolute_max_normalized','maximum_absolute')]:
                                compare(values['error'][key],row['metrics'][metric],metric)
                            emit(fid,index,values)
            finally:
                guard.close()
            guard_evidence = guard.evidence()
            require(not guard.calls and not guard.forbidden_attempts and not torch.cuda.is_initialized(), 'No learned/CPU/CUDA forward permitted in the fixed decoder')
    reread = Counter()
    with gzip.open(pending,'rt') as stream:
        for line in stream:
            reread[json.loads(line)['fit_item_id']] += 1
    require(reread == counts and len(counts) == 10 and sum(counts.values()) == 6128, 'Every10fit component target population must roundtrip')
    pending.replace(output)
    save(report_path,dict(status='ten_component_field_populations_reduced_to_task_specific_scalars_no_predictions_saved',
        created_unix=time.time(),producer=binding(Path(__file__)),helper=binding(Path(__file__).with_name('pressure_scalar_diagnostics.py')),
        source_reports=list(sources.values()),prediction_capture_identities=list(captures.values()),
        model_descriptions=descriptions,case_contexts=contexts,output_records_by_fit=dict(counts),output_records=sum(counts.values()),
        accepted_metric_comparisons=dict(checks),
        scalar_case_payload=dict(**binding(output),bytes=output.stat().st_size,format='gzip UTF8 JSON Lines;case/stage scalar statistics only'),
        CPU_fixed_decoder_forward_guard=guard_evidence,
        definitions=dict(errors='Original task/output units, absolute RMS/MAE/max/quantiles, signed bias, L2 target/floor and relative L2',
            pressure='Mean-free field error distinct from cell-zero gauged error and equation residual',
            local_synthetic='Physical mask only;12cubed synthetic patterns have no assigned cells/metre',
            sampler='Middle/end XYZ local velocity-equivalent targets;these are not two physical simulation outputs',
            q='Normalized future request only; original filtered-pressure objective remains separate in its accepted scalar report',
            applicability='Full-state pressure/velocity/tracer conservation,physical lead and own-history drift do not follow from local component targets'),
        fixed_scoring_wall_seconds=time.monotonic()-began,new_model_forwards=0,CPU_neural_forwards=0,
        model_weight_payloads_opened=0,new_reference_generation=0,field_copies_saved=0,prediction_files_deleted=0,
        deletion_paused=True,native_mutations=0,full_portfolio_complete=False))
    print('Task-specific scalar populations:',dict(counts),'records',sum(counts.values()),'checks',sum(checks.values()),';no fields saved or deleted.',flush=True)


if __name__ == '__main__':
    run()
