"""Compare the fused decoder with the unchanged tensor implementation."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest as sha
import statistics
import time
import numpy as np
import torch
from displacement_readout import convex_field_readout as reference
from vector_field_readout import convex_field_readout as fast_candidate
from fused_field_readout import convex_field_readout as old_candidate
from training_displacement_teacher import displacement_labels
from transport_teacher_support import carrier
HERE=Path(__file__).resolve().parent
BASE=ROOT/'runs/full-ai-next-priorities-v1'
DOCS=ROOT/'docs/plans/full-ai-tight-targets-v1'


TIMINGS=[]
CALL_INDEX=0


def candidate(fields,shifts):
    global CALL_INDEX
    result=fast_candidate(fields,shifts)
    index=CALL_INDEX;CALL_INDEX+=1
    # Original frozen qualification places its 216 stress fixtures first,
    # followed by 64 source-bound captured training cases. Only the latter
    # form this component latency population.
    if 216<=index<280:
        expected=old_candidate(fields,shifts)
        if not torch.equal(expected,result):raise ValueError('Existing compiled readout differs')
        row=dict(case_index=index,carrier_shape_zyx=list(fields.shape[2:]),baseline_ms=[],candidate_ms=[])
        functions={'baseline_ms':old_candidate,'candidate_ms':fast_candidate}
        for repeat in range(5):
            order=('baseline_ms','candidate_ms')if(index+repeat)%2==0 else('candidate_ms','baseline_ms')
            for name in order:
                tick=time.perf_counter();actual=functions[name](fields,shifts)
                elapsed=(time.perf_counter()-tick)*1000
                if not torch.equal(actual,result):raise ValueError('Timed readout changed exact values')
                row[name].append(elapsed)
        TIMINGS.append(row)
    return result


def summaries():
    result=[]
    for regime in ('small','32cubed'):
        selected=[r for r in TIMINGS if(r['carrier_shape_zyx']==[33,33,33])==(regime=='32cubed')]
        for variant in ('baseline_ms','candidate_ms'):
            values=[value for row in selected for value in row[variant]]
            result.append(dict(regime=regime,variant=variant,observations=len(values),
                median_ms=statistics.median(values),maximum_ms=max(values)))
    return result


def run(args):
    if args.output.exists():raise ValueError('Fresh fused readout qualification required')
    protocol_path=DOCS/'vector-readout-qualification-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Registered readout source changed')
    os.sched_setaffinity(0,{args.cpu});torch.set_num_threads(1)
    rng=np.random.Generator(np.random.PCG64(protocol['seed']));cases=[];sources={}
    for shape in protocol['carrier_shapes_zyx']:
        for pattern in protocol['field_patterns']:
            values=rng.normal(size=(1,4,*shape)).astype('f4')
            if pattern=='zero':values.fill(0)
            if pattern=='constant':values.fill(.25)
            if pattern=='ramp':values=np.linspace(-1,1,values.size,dtype='f4').reshape(values.shape)
            if pattern=='tiny':values=values*1e-40
            if pattern=='wide':values=values*1e30
            for kind in protocol['shift_patterns']:
                shifts=rng.uniform(-1,1,size=(1,4,3,*shape)).astype('f8'if kind in ('random64','subnormal')else'f4')
                if kind=='zero':shifts.fill(0)
                if kind=='integer_signed':shifts=np.sign(shifts)
                if kind=='small':shifts=shifts*1e-8
                if kind=='subnormal':shifts=shifts*1e-310
                cases.append((torch.from_numpy(values.copy()),torch.from_numpy(shifts.copy()),dict(kind='fixture',shape=shape,pattern=pattern,shift_kind=kind)))
    for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
        if row['split']!='train':continue
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:raise ValueError('Training receipt changed')
        receipt=read(path);scene=receipt['scene'];sources[str(path.relative_to(ROOT))]=sha(path)
        for start in protocol['starts']:
            spec=receipt['states'][start];source=path.parent/spec['path']
            if sha(source)!=spec['sha256']:raise ValueError('Native training field changed')
            sources[str(source.relative_to(ROOT))]=sha(source)
            with np.load(source,allow_pickle=False)as a:fields={n:a[n].copy()for n in ('vx','vy','vz','tracer')}
            value=carrier(fields,scene['counts']);shift,_=displacement_labels(value[:,:3],scene['counts'],scene['cells_per_metre']/60)
            cases.append((value,shift,dict(kind='captured',identifier=row['identifier'],start=start)))
    rows=[]
    with torch.inference_mode():
        for fields,shifts,identity in cases:
            expected=reference(fields,shifts);actual=candidate(fields,shifts)
            equal=torch.equal(actual,expected)
            if not equal:raise ValueError('Fused readout differs numerically at float32 precision: '+str(identity))
            bitwise=np.array_equal(actual.numpy().view('u1'),expected.numpy().view('u1'))
            if float(actual.min())<float(fields.min())or float(actual.max())>float(fields.max()):raise ValueError('Fused output left donor range')
            if bool((fields==0).all())and bool((actual!=0).any()):raise ValueError('Zero field changed')
            if bool((fields==.25).all())and not bool((actual==.25).all()):raise ValueError('Constant field changed')
            rows.append(dict(**identity,float32_values_exact=True,bytes_exact=bitwise))
    if len(rows)!=protocol['cases']:raise ValueError('Registered probe population differs')
    errors=[]
    for label,which in (('nonfinite_field','field'),('nonfinite_shift','shift'),('out_of_support','support')):
        value=torch.ones(1,4,3,3,3);shift=torch.zeros(1,4,3,3,3,3)
        if which=='field':value[0,0,0,0,0]=float('nan')
        elif which=='shift':shift[0,0,0,0,0,0]=float('inf')
        else:shift[0,0,0,0,0,0]=1.01
        observed=[]
        for fn in (reference,candidate):
            try:fn(value,shift)
            except (ValueError,FloatingPointError)as error:observed.append(type(error).__name__)
        if len(observed)!=2 or len(set(observed))!=1:raise ValueError('Finite/support guard behavior differs')
        errors.append(dict(case=label,error_class=observed[0]))
    write(args.output,dict(status='passed',rows=rows,guard_cases=errors,source_evidence=sources,source_sha256=protocol['source_sha256'],
        protocol_sha256=sha(protocol_path),build_receipt_sha256=sha(ROOT/'runs/full-ai-tight-targets-v1/vector-field-readout-build-01/receipt.json'),
        float32_values_exact_every_case=True,all_bytes_exact=all(r['bytes_exact']for r in rows),optimizer_updates=0,
        no_model_weights_changed=True,CPU_latency_claim=False,full_model_qualification_required=True,reserved_scenes_consumed=False,
        component_timings=TIMINGS,component_timing_summaries=summaries(),
        CPU_component_latency_only=True,CPU=args.cpu,threads=1,host_ISA_specific_binary=True))
    print('Qualified fused readout:',len(rows),'exact float32 value cases;',len(errors),'guard cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cpu',type=int,choices=(6,23),default=23);run(parser.parse_args())
