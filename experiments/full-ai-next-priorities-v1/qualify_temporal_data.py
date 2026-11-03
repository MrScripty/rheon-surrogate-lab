"""Qualify complete causal state and real binary restart at the Python boundary."""
import os
import time
import numpy as np
import torch
from study_support import ROOT,BASE,DOCS,read,write,sha
from native_client import NativeClient,FIELDS

def run():
    output=BASE/'temporal-data-qualification-01'
    if output.exists():raise ValueError('Fresh data qualification required')
    output.mkdir(parents=True)
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    reservation=read(DOCS/'temporal-scene-reservation.json')
    rows=[]
    for index,counts in enumerate(([16,16,16],[9,11,11])):
        source=next(r for r in reservation['rows']if r['counts']==list(counts)and r['split']=='train')
        scene={k:v for k,v in source.items()if k not in ('identifier','split')}
        scene['steps']=16
        a=NativeClient(scene,output/f'original-{index}')
        b=NativeClient(scene,output/f'restarted-{index}')
        try:
            _,initial=a.inspect()
            for _ in range(4):
                if not a.advance()['frame']['advanced']:raise ValueError('Reference rejected during restart prefix')
            packet,saved=a.inspect();b.restore(packet['step'],saved)
            comparisons=[]
            for _ in range(8):
                fa=a.advance();fb=b.advance()
                if not fa['frame']['advanced']or not fb['frame']['advanced']:raise ValueError('Restart reference rejected')
                pa,sa=a.inspect();pb,sb=b.inspect()
                if pa['step']!=pb['step']:raise ValueError('Restart clock mismatch')
                equality={name:np.array_equal(sa[name],sb[name])for name in FIELDS}
                if not all(equality.values()):raise ValueError('Native restart changed subsequent fields')
                if not np.array_equal(a.last_request['rhs'],b.last_request['rhs']):raise ValueError('Restart pressure equation changed')
                comparisons.append(dict(step=pa['step'],time_s=pa['time_s'],bitwise_fields=equality,rhs_bitwise_equal=True))
            rows.append(dict(counts_xyz=counts,cells_per_metre=scene['cells_per_metre'],restart_step=4,
                subsequent_steps=8,comparisons=comparisons,initial_fields={name:dict(dtype=v.dtype.str,shape=list(v.shape))for name,v in initial.items()}))
        finally:
            ar=a.close();br=b.close()
        rows[-1]['native_processes']=[ar,br]
    engine=BASE/'temporal-build-02/rheon/src/diagnostic/engine.rs'
    write(output/'qualification.json',dict(status='passed',rows=rows,optimizer_updates=0,
        restart_tolerance='All field bytes and future RHS exactly equal at every compared step',
        contract=dict(state_fields=['pressure_f64','vx_f32','vy_f32','vz_f32','tracer_f32','physical_step_index'],
            static_context=['scene geometry and cell spacing','rho=1000kg/m3','mu=0.001Pa.s','dt=1/60s',
                'closed boundaries','complete declared forcing parameters and seed'],
            hidden_history='No additional dynamic history: pressure and viscosity workspaces are resettable scratch; verified by native and wire restart tests',
            pressure_phase='Pressure solved during the transition ending at the state timestamp',
            velocities='Post-projection face velocities; X fastest, then Y, then Z',
            tracer='Cell-centered tracer after advection in the transition ending at the state timestamp',
            initial_state_included=True,future_reference_input=False),
        source_sha256=sha(__file__),native_client_sha256=sha(BASE.parents[1]/'experiments/full-ai-next-priorities-v1/native_client.py'),
        native_engine_sha256=sha(engine),build_receipt_sha256=sha(BASE/'temporal-build-02/build-receipt.json'),
        reservation_sha256=sha(DOCS/'temporal-scene-reservation.json'),application_promotion=False))
    print('Qualified eight future full native states and RHS bitwise after restart on both grids',flush=True)

if __name__=='__main__':run()
