"""Capture deterministic student initializations from the completed full teacher."""
import argparse
import os
from pathlib import Path
from v4_support import BASE,load,write,sha

def run(args):
    if args.output.exists():raise ValueError('Fresh student initialization required')
    os.sched_setaffinity(0,{22})
    state,_,_=load()
    from student_slice import slice_width
    inventory=state.data.helpers().state_inventory(state.network)
    network,slices=slice_width(state.network,*(tuple(map(int,args.width.split('/')))))
    args.output.mkdir(parents=True)
    result=state.data.helpers().save_weights(network,args.output/'weights.npz')
    if inventory!=state.data.helpers().state_inventory(state.network):raise ValueError('Original full teacher changed')
    write(args.output/'initialization.json',dict(status='qualified_initialization',source_entry=state.entry,
        teacher_inventory=inventory,weights_sha256=sha(args.output/'weights.npz'),inventory=result,
        parameters=sum(p.numel()for p in network.parameters()),width_fraction=list(map(int,args.width.split('/'))),slices=slices,
        source_sha256=sha(__file__),slicing_source_sha256=sha(Path(__file__).with_name('student_slice.py')),optimizer_updates=0))
    print('Captured student initialization',args.width,sum(p.numel()for p in network.parameters()),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--width',choices=['3/4','1/2'],required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
