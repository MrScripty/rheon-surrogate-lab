"""Package retained C106/C107 scalar diagnostics without executing any model.

Original capture JSON is retained byte for byte. Each measured scalar points
back to its original JSON value; unmeasured latency remains explicitly missing.
Native project admission/rendering is a separate gate.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/plans/model-portfolio-migration-v1'
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok-measurements'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(output):
    if output.exists() or not output.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1'):
        raise ValueError('Fresh task-owned measurement output required')
    sys.path.insert(0,str(SOURCE))
    from retained_measurements import MeasurementTable, digest, encode
    common_path=DOCS/'reports/pressure-common-score-01.json'
    common=json.loads(common_path.read_text())
    manifest_path=ROOT/'runs/water-early-pressure-v1/control/project/inputs/validation_splits.json'
    manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw)
    if manifest['schema']!='rheon-water-learning-view-v1' or manifest['population']!='validation':
        raise ValueError('Original consumed pressure packet required')
    sources=[];captures={};receipts={}
    def add(identity,label,raw):
        sources.append(dict(id=identity,label=label,kind='json',bytes=len(raw),sha256=sha(raw),
                            data_base64=base64.b64encode(raw).decode('ascii')))
    for name in ('C106','C107'):
        receipt=common['models'][name]['result'];path=ROOT/receipt['path'];raw=path.read_bytes()
        if sha(raw)!=receipt['sha256']:
            raise ValueError('Retained scalar result identity changed')
        value=json.loads(raw)
        if (value['identifier']!=name or value['status']!='completed_captured_pressure_GPU_accuracy' or value['cases']!=128
                or value['CPU_model_forward'] or value['neural_module_forward_calls']['cpu']
                or not value['architecture_unchanged'] or not value['learned_parameters_unchanged']
                or value['latency_claim']):
            raise ValueError('Expected complete unchanged GPU accuracy capture without a timing claim')
        if (value['pressure_metric']!='Mean-free pressure error L2 / max(mean-free reference L2, 1e-30); absolute RMS and spatial maximum retained.'
                or value['equation_metric']!='Captured full_operator pressure residual L2 / max(RHS L2, 1e-30).'):
            raise ValueError('Retained metric definitions differ from this packet producer')
        archive=path.parent/value['output_archive']['path']
        if sha(archive.read_bytes())!=value['output_archive']['sha256']:
            raise ValueError('Retained pressure output archive changed')
        add(name,'Original '+name+' GPU pressure accuracy capture',raw)
        captures[name]=value;receipts[name]=receipt
    metric_specs=[
        ('equation_relative_l2','Pressure equation residual (relative L2)','1',
         'L2(A p - rhs) / max(L2(rhs), 1e-30), using the captured full_operator.'),
        ('pressure_mean_free_relative_l2','Mean-free pressure field error (relative L2)','1',
         'L2((p - reference) - mean(p - reference)) / max(L2(reference - mean(reference)), 1e-30).'),
        ('pressure_absolute_rms','Mean-free pressure absolute RMS error','captured pressure units',
         'Root mean square of (p - reference) - mean(p - reference). No unit conversion.'),
        ('pressure_absolute_max','Mean-free pressure maximum absolute error','captured pressure units',
         'Spatial maximum of abs((p - reference) - mean(p - reference)). No unit conversion.'),
        ('pressure_call_ms','Complete pressure-call latency','ms',
         'Qualified complete pressure-call wall time, with its device and boundary. These retained accuracy captures provide no such measurement.')]
    definitions=[dict(id=i,label=label,units=units,description=description,
                      source_kind='retained capture values; no new metric computation')
                 for i,label,units,description in metric_specs]
    case_keys=('trajectory_id','parent','native_frame','counts_xyz','cell_size_m',
               'input_sha256','reference_pressure_sha256','reference_rhs_sha256')
    bindings=[];cases=[]
    for index,(left,right) in enumerate(zip(captures['C106']['rows'],captures['C107']['rows'],strict=True)):
        binding={k:left[k] for k in case_keys}
        if binding!={k:right[k] for k in case_keys}:
            raise ValueError('Subjects do not have identical input/reference cases')
        parent=manifest['parents'][left['parent']]
        frame=next(v for v in parent['frames'] if v['index']==left['native_frame'])
        if (parent['entry']['trajectory_id']!=left['trajectory_id'] or parent['entry']['split']!='validation'
                or parent['counts_xyz']!=left['counts_xyz'] or parent['cell_size_m']!=left['cell_size_m']):
            raise ValueError('Case metadata differs from its original packet')
        for name,value in [('dt',parent['entry']['dt_s']),('rho',parent['density_kg_m3']),('cell_size_m',parent['cell_size_m'])]:
            if sha(struct.pack('<d',value))!=left['input_sha256'][name]:
                raise ValueError('Physical scalar metadata differs from captured input bytes')
        for name,target in [('base',left['input_sha256']['base']),('native_input_residual',left['input_sha256']['native_input_residual']),
                            ('reference',left['reference_pressure_sha256']),('rhs',left['reference_rhs_sha256'])]:
            if frame['sha256'][name]!=target:
                raise ValueError('Source frame identity differs')
        dims=dict(grid_x=left['counts_xyz'][0],grid_y=left['counts_xyz'][1],grid_z=left['counts_xyz'][2],
                  cells_per_metre=1./left['cell_size_m'],input_dt_s=parent['entry']['dt_s'],
                  fluid_density_kg_m3=parent['density_kg_m3'],native_frame=left['native_frame'],trajectory_id=left['trajectory_id'])
        for key in ('equation_rhs_l2','pressure_reference_mean_free_l2','equation_floor_active','pressure_floor_active'):
            if left['metrics'][key]!=right['metrics'][key]:
                raise ValueError('Reference denominator or normalization-floor metadata differs')
            dims[key]=left['metrics'][key]
        case_id=f'case-{index:03d}'
        bindings.append(dict(id=case_id,dimensions=dims,original_case=binding))
        cases.append(dict(id=case_id,dimensions=dims,reference=dict(source_id='context',pointer='/cases/'+str(index))))
    if len(bindings)!=128:
        raise ValueError('Complete original case denominator required')
    scope=dict(id='old-pressure-consumed-128',label='Original consumed pressure requests · GPU accuracy',population='consumed',
        task='pressure proposal, not complete simulation advancement',
        reference_epoch=dict(captured_packet_manifest_sha256=sha(manifest_raw),new_Rheon_adopted=False),
        case_binding_sha256=digest(bindings),case_count=128,
        acceptance='Diagnostic observations on consumed cases; no independent validation or application acceptance.',
        interpretation='Equation residual and mean-free pressure error are distinct. Native frame indices are not reconstructed physical timestamps.')
    context=dict(scope=scope,metrics=definitions,cases=bindings,capture_receipts=receipts,
                 original_packet_manifest_sha256=sha(manifest_raw),common_score_report_sha256=sha(common_path.read_bytes()))
    add('context','Exact comparison, metric and original case definitions',encode(context))
    add('packet','Original consumed pressure packet manifest',manifest_raw)
    add('common-score','Retained independent common-score qualification',common_path.read_bytes())
    def reference(pointer,value):return dict(source_id='context',pointer=pointer,sha256=digest(value))
    dimensions=[]
    for identity,label,kind,units in [
        ('grid_x','Domain X cells','integer','cells'),('grid_y','Domain Y cells','integer','cells'),
        ('grid_z','Domain Z cells','integer','cells'),('cells_per_metre','Grid density','number','cells/m'),
        ('input_dt_s','Explicit input timestep','number','s'),('fluid_density_kg_m3','Fluid density','number','kg/m^3'),
        ('native_frame','Captured native frame index','integer',None),
        ('trajectory_id','Original trajectory identity','string',None),
        ('equation_rhs_l2','Equation RHS L2 denominator','number','captured RHS units'),
        ('pressure_reference_mean_free_l2','Mean-free reference L2 denominator','number','captured pressure units'),
        ('equation_floor_active','Equation normalization floor active','boolean',None),
        ('pressure_floor_active','Pressure normalization floor active','boolean',None)]:
        dimensions.append(dict(id=identity,label=label,type=kind,units=units,scope='case'))
    subject_fields=[('inference_device','Inference device','string'),('gpu_model','GPU model','string'),
                    ('network_dtype','Network precision','string'),('pressure_dtype','Pressure precision','string'),
                    ('tf32','TF32 enabled','boolean'),('autocast','Autocast enabled','boolean'),
                    ('torch_version','Torch version','string'),('cuda_version','CUDA version','string'),
                    ('weights_sha256','Exact retained weight identity','string')]
    for identity,label,kind in subject_fields:
        dimensions.append(dict(id=identity,label=label,type=kind,units=None,scope='subject'))
    subjects=[dict(id=name,label=name,dimensions={k:captures[name][k] for k,_label,_kind in subject_fields},
                   reference=dict(source_id=name,pointer='')) for name in ('C106','C107')]
    observations=[]
    for name in ('C106','C107'):
        for index,row in enumerate(captures[name]['rows']):
            for definition in definitions:
                metric=definition['id'];timing=metric=='pressure_call_ms'
                observations.append(dict(id=name+f'-case-{index:03d}-'+metric,subject_id=name,case_id=f'case-{index:03d}',
                    metric_id=metric,state='unavailable' if timing else 'measured',
                    value=None if timing else row['metrics'][metric],
                    reason='Accuracy capture explicitly has latency_claim=false; no qualified call timing.' if timing else None,
                    dimensions={},reference=dict(source_id=name,pointer='/latency_claim' if timing else f'/rows/{index}/metrics/{metric}')))
    packet=dict(format='retained_measurements_v1',schema_version=1,title='C106 and C107 · original pressure diagnostics',
        scope={**{k:scope[k] for k in ('id','label','population')},'definition':reference('/scope',scope)},
        dimensions=dimensions,subjects=subjects,cases=cases,
        metrics=[{**{k:m[k] for k in ('id','label','units')},'definition':reference('/metrics/'+str(i),m)} for i,m in enumerate(definitions)],
        observations=observations,sources=sources)
    raw=encode(packet);table=MeasurementTable(raw)
    measured=0
    for name in ('C106','C107'):
        for index,row in enumerate(captures[name]['rows']):
            for metric,*_ in metric_specs[:4]:
                actual=table.inspect(name+f'-case-{index:03d}-'+metric)['observation']['value']
                if struct.pack('!d',actual)!=struct.pack('!d',row['metrics'][metric]):
                    raise ValueError('A retained binary64 scalar changed during packaging')
                measured+=1
    output.mkdir();(output/'measurements.json').write_bytes(raw)
    if any(name=='torch' or name.startswith('torch.') or name=='numpy' for name in sys.modules):
        raise ValueError('No numerical/model runtime belongs in scalar packaging')
    data_path=(output/'measurements.json').relative_to(ROOT).as_posix()
    declaration=dict(schema_version=1,id=sha(data_path.encode()+bytes.fromhex(sha(raw)))[:32],
        name='Retained C106 C107 pressure measurements',
        data=[dict(id='pressure-measurements',label='Exact retained scalar diagnostics and explicit timing gaps',
                   path=data_path,sha256=sha(raw))],recipes=[],evidence=[],requirements=[])
    from projects import manifest as source_manifest
    source_manifest(declaration)
    manifest_bytes=(json.dumps(declaration,indent=2)+'\n').encode()
    (output/'source-project.json').write_bytes(manifest_bytes)
    report=dict(schema='rheon.native-retained-measurements-preparation.v1',status='qualified_scalar_packet_not_yet_native_imported',
        packet_sha256=sha(raw),packet_bytes=len(raw),subjects=2,cases=128,measured_values_exact=measured,
        explicitly_unavailable_timings=256,source_capture_receipts=receipts,
        pressure_summary=table.query('pressure_mean_free_relative_l2')['summary'],
        equation_summary=table.query('equation_relative_l2')['summary'],
        timing_summary=table.query('pressure_call_ms')['summary'],
        grid_density_values=sorted({v['dimensions']['cells_per_metre'] for v in cases}),
        density_and_grid_groups=sorted({(v['dimensions']['cells_per_metre'],v['dimensions']['grid_x'],v['dimensions']['grid_y'],v['dimensions']['grid_z']) for v in cases}),
        input_dt_values=sorted({v['dimensions']['input_dt_s'] for v in cases}),
        codec_sha256=sha((SOURCE/'retained_measurements.py').read_bytes()),runner_sha256=sha(Path(__file__).read_bytes()),
        source_manifest_sha256=sha(manifest_bytes),source_reference_id='pressure-measurements',
        new_inference=False,new_training=False,numeric_runtime_imported=False,native_project_imported=False,native_chart_qualified=False,
        interpretation='Exact retained scalars and case/input identities; no new scoring, timing, physics or scientific acceptance claim.')
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','packet_bytes','measured_values_exact','explicitly_unavailable_timings','input_dt_values')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
