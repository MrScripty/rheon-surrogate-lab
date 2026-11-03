"""Cadence-specific temporal pilot measurements, separate from 60 Hz catalogs."""
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import statistics

FIELDS=('equation','pressure','velocity','tracer')


def enrich_temporal_pilots(data,base,lab,load):
    base,lab=Path(base),Path(lab);docs=base/'full-ai-data-diversity-v1';root=lab/'runs/full-ai-data-diversity-v1'
    if not (root/'temporal-frozen-GPU-benchmark-01/results.json').exists():return
    hashes={}
    def sha(path):
        path=Path(path)
        if path not in hashes:
            with path.open('rb') as stream:hashes[path]=hashlib.file_digest(stream,'sha256').hexdigest()
        return hashes[path]
    def source(path):return os.path.relpath(path,base)
    def witness(value):
        for key in ('source_sha256','source_evidence','reference_sources'):
            for name,expected in (value.get(key) or {}).items():
                if isinstance(expected,str) and len(expected)==64 and sha(lab/name)!=expected:
                    raise ValueError('Temporal pilot evidence changed: '+name)
    def read(path,expected=None):
        value=load(Path(path),expected);witness(value)
        if value.get('new_parameter_fits',0) or value.get('residual_optimizer_updates',0) or value.get('cpu_model_forward',False) or value.get('reserved52_references_opened',0):
            raise ValueError('Temporal pilot unexpectedly fitted or opened reserved data')
        return value
    result=dict(captures=[],points=[],readiness=None)
    captures=[('frozen01','Historical fine/grouped · original pressure history','temporal-frozen-GPU-benchmark-01','temporal-frozen-GPU-benchmark-protocol-01.json'),
        ('frozen02','Historical fine/grouped · pinned transfer','temporal-frozen-GPU-benchmark-02','temporal-frozen-GPU-benchmark-protocol-02.json'),
        ('frozen03','Repaired fine baseline · retained history','temporal-frozen-GPU-benchmark-03','temporal-frozen-GPU-benchmark-protocol-03.json'),
        ('residual01','Untrained temporal residual · eager/graph','temporal-residual-GPU-benchmark-01','temporal-residual-GPU-benchmark-protocol-01.json'),
        ('residual02','Untrained temporal residual · paired gather graphs','temporal-residual-GPU-benchmark-02','temporal-residual-GPU-benchmark-protocol-02.json'),
        ('shared01','Untrained shared encoder · original','shared-temporal-GPU-benchmark-01','shared-temporal-GPU-benchmark-protocol-01.json'),
        ('shared02','Untrained shared encoder · paired force graphs','shared-temporal-GPU-benchmark-02','shared-temporal-GPU-benchmark-protocol-02.json'),
        ('shared03','Untrained shared encoder · stationary-prefix repair','shared-temporal-GPU-benchmark-03','shared-temporal-GPU-benchmark-protocol-03.json')]
    original_shared={}
    for identifier,label,folder,protocol in captures:
        path=root/folder/'results.json'
        if not path.exists():continue
        value=read(path)
        if not value['status'].startswith('completed_'):continue
        reg=read(docs/protocol,value['protocol_sha256'])
        qualification_path=lab/reg['qualification'];qualification=read(qualification_path)
        if not qualification['status'].startswith(('completed_','passed_')):raise ValueError('Pilot qualification is not terminal')
        qualified=defaultdict(list)
        for row in qualification['rows']:
            key=(row['identifier'],row['start_fine_step'],row['output_steps_per_s'],row['K'],row.get('mode','shared'),row.get('hidden_width'))
            qualified[key].append(row)
        if identifier=='shared02':
            evidence=lab/reg['native_accuracy_evidence']
            if source(evidence)!=original_shared['source'] or sha(evidence)!=original_shared['sha256']:
                raise ValueError('Paired-force inherited native metrics differ')
        result['captures'].append(dict(id=identifier,label=label,source=source(path),qualification_source=source(qualification_path),
            status=value['status'],scope=value['scope'],event_scope=value.get('event_scope','CUDA events span the complete asynchronous transaction, including pipeline gaps; not kernel-only timing.'),
            untrained=identifier.startswith(('residual','shared')),historical=identifier in ('frozen01','frozen02'),
            resource_note=reg.get('scope',''),gpu_model=value.get('gpu_model',value.get('GPU'))))
        for row in value['rows']:
            k=row['K'];cadence=row['output_steps_per_s'];mode=row.get('mode','shared');width=row.get('hidden_width')
            key=(row['identifier'],row['start_fine_step'],cadence,k,mode,width)
            if not row.get('every_output_hash_matches_qualification',row.get('complete_output_hashes_match_qualification',False)):
                raise ValueError('Actual pilot timed output qualification missing')
            if 'native_metrics_per_output' in row:
                metrics=row['native_metrics_per_output']
                # Shared metrics were either recomputed, or inherited from the
                # exact original capture explicitly closed by report 02.
                if identifier=='shared02' and metrics!=original_shared['metrics'][key]:
                    raise ValueError('Paired shared native errors differ')
                if identifier=='shared01':
                    original_shared.setdefault('metrics',{})[key]=metrics
            else:
                scored=qualified[key]
                scored.sort(key=lambda r:r['lead'])
                if [r['lead'] for r in scored]!=list(range(1,k+1)):raise ValueError('Missing intermediate pilot output')
                metrics=[r['metrics'] for r in scored]
            if len(metrics)!=k:raise ValueError('Temporal pilot metric output count differs')
            errors={f:max(m['relative'][f] for m in metrics) for f in FIELDS}
            if not all(math.isfinite(e) and e>=0 for e in errors.values()):raise ValueError('Nonfinite pilot error')
            wall=row['block_wall_ms'];events=row['pipeline_cuda_event_span_ms']
            if len(wall)!=reg['repetitions'] or len(events)!=len(wall) or any(t<=0 or not math.isfinite(t) for t in wall+events):raise ValueError('Incomplete pilot timing repeats')
            if statistics.median(wall)!=row['median_block_ms'] or max(wall)!=row['maximum_block_ms']:raise ValueError('Pilot wall sample statistics differ')
            if not math.isclose(row['block_duration_s'],k/cadence,rel_tol=1e-12):raise ValueError('Pilot physical block duration differs')
            result['points'].append(dict(capture=identifier,identifier=row['identifier'],split=row['split'],counts_xyz=row['counts_xyz'],
                cells_per_metre=row['cells_per_metre'],start_fine_step=row['start_fine_step'],cadence=cadence,K=k,
                mode=mode,backend=row.get('backend','graph'),hidden_width=width,neural_parameters=row.get('neural_parameters',12732 if identifier.startswith('residual') else None),
                internal_advances=row.get('actual_internal_advances'),coarse_endpoint_transports=row.get('coarse_endpoint_transports'),
                encoder_calls=row.get('shared_spatial_encoder_calls'),pressure_projections=row.get('learned_pressure_projections',row.get('actual_learned_pressure_projections')),
                field_worst=errors,accuracy_pass=row['all_accuracy_pass'],scored_outputs=k,wall_samples_ms=wall,event_samples_ms=events,
                median_block_ms=row['median_block_ms'],maximum_block_ms=row['maximum_block_ms'],source=source(path),
                accuracy_source=source(lab/reg['native_accuracy_evidence']) if identifier=='shared02' else source(path) if 'native_metrics_per_output' in row else source(qualification_path)))
        if identifier=='shared01':original_shared.update(source=source(path),sha256=sha(path))
    # Readiness is reference/training-interface evidence, not a completed fit or
    # a 36-scene model score. Transient training archives are not imported.
    ready_path=docs/'shared-temporal-captured-training-readiness-01.json'
    if ready_path.exists():
        ready=load(ready_path)
        audit_path=docs/'long-reference-expansion-audit-progress-06.json';audit=load(audit_path)
        load(lab/audit['qualification'],audit['qualification_sha256'])
        if ready['new_parameter_fits'] or audit['new_parameter_fits'] or ready['reserved52_references_opened'] or audit['reserved52_references_opened']:
            raise ValueError('Readiness incorrectly establishes a trained model')
        result['readiness']=dict(source=source(ready_path),audit_source=source(audit_path),training_masters=24,development_masters=12,
            fine_steps_per_second=240,duration_s=20,finite_f32_recoveries=audit['explicit_finiteF32_recoveries'],temporal_convergence_proven=audit['temporal_convergence_proven'],
            training_windows=ready['training_windows'],physical_outputs_per_width=ready['physical_outputs_per_width'],managed_fitting_entrypoint_ready=ready['managed_fitting_entrypoint_ready'],new_parameter_fits=0,reserved_opened=0)
    from temporal_successor_chart import enrich_temporal_successors
    enrich_temporal_successors(result,base,lab,load)
    data['temporal_pilots']=result
