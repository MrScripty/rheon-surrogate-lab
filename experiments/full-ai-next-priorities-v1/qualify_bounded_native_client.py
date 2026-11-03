"""Qualify exact native transport and eight-step restart under bounded I/O."""
import argparse
import os
from pathlib import Path
import signal
import numpy as np
import torch
from bounded_native_client import BoundedNativeClient
from native_client import FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh native transport qualification required')
    protocol_path=DOCS/'bounded-native-client-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered transport source changed')
    for relative,digest in protocol['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Qualified native build changed')
    bank_path=BASE/'temporal-development-240-01/bank.json'
    if sha(bank_path)!=protocol['bank_sha256']:raise ValueError('Registered scene bank changed')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    def stop(*_):raise InterruptedError('Transport qualification interrupted; reaping native child')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    rows=[];sources={};bank=read(bank_path)
    for identifier in protocol['scenes']:
        row=next(r for r in bank['trajectories']if r['identifier']==identifier)
        if row['split']!='development':raise ValueError('Consumed development scenes only')
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:raise ValueError('Reference receipt changed')
        receipt=read(path);scene=dict(receipt['scene'],steps=16);sources[str(path.relative_to(ROOT))]=sha(path)
        states={}
        for step in (0,*range(8,17)):
            spec=receipt['states'][step];source=path.parent/spec['path']
            if sha(source)!=spec['sha256']:raise ValueError('Saved reference state changed')
            sources[str(source.relative_to(ROOT))]=sha(source)
            with np.load(source,allow_pickle=False)as a:states[step]={n:a[n].copy()for n in FIELDS}
        client=BoundedNativeClient(scene,args.output/identifier);compared=0
        def compare(step):
            nonlocal compared
            packet,actual=client.inspect()
            if packet['step']!=step or packet['time_s']!=step/60:raise ValueError('Native restart clock differs')
            for name in FIELDS:
                if not np.array_equal(actual[name].view('u1'),states[step][name].view('u1')):
                    raise ValueError('Native physical bytes differ: '+name)
            compared+=1
        try:
            compare(0);client.restore(8,states[8]);compare(8)
            for step in range(9,17):
                frame=client.advance(method='spectral')
                if not frame['frame']['advanced']:raise ValueError('Native qualification reference rejected')
                compare(step)
            custody=client.close()
        finally:
            if client.last_custody is None:client.close()
        if not custody['reaped']or custody['returncode']!=0 or not custody['graceful_protocol_complete']:
            raise ValueError('Native child must complete and reap')
        rows.append(dict(identifier=identifier,counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
            physical_states_compared=compared,eight_subsequent_steps_bitwise_exact=True,native_process=custody))
        print('Qualified bounded native transport and eight-step restart:',identifier,flush=True)
    write(args.output/'receipt.json',dict(status='passed',rows=rows,source_sha256=protocol['source_sha256'],
        source_evidence=sources,protocol_sha256=sha(protocol_path),
        optimizer_updates=0,performance_claim=False,application_physical_acceptance_claim=False,
        native_reference_solves='Offline qualification only; never a candidate inference component',reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
