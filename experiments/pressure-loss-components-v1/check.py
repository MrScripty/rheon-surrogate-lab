"""Literal geometry and analytic chain-rule oracle for the diagnostic helpers."""
import argparse
from pathlib import Path
import tempfile
import torch
import numpy as np
from measure import geometry
from common import write,sha


def check(output):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='loss-component-oracle-')as temporary:
        paths=[]
        for index,value in enumerate([[1,0],[-1,0],[0,2],[0,0]]):
            p=Path(temporary)/(str(index)+'.npy');np.save(p,np.array(value,dtype='<f4'));paths.append(p)
        g=geometry(paths)
        assert g['norms']==[1,1,2,0]
        assert g['dot_matrix']==[[1,-1,0,0],[-1,1,0,0],[0,0,4,0],[0,0,0,0]]
        assert g['cosine_matrix']==[[1,-1,0,None],[-1,1,0,None],[0,0,1,None],[None,None,None,None]]
    w=torch.tensor(2.,dtype=torch.float32,requires_grad=True)
    terms=[2*w*w,3*w,w*w];parts=[float(torch.autograd.grad(v,w,retain_graph=True)[0])for v in terms]
    joint=float(torch.autograd.grad(sum(terms),w)[0])
    assert parts==[8,3,4]and joint==15 and sum(parts)==joint
    assert not torch.cuda.is_initialized()
    write(output,dict(status='passed',oracle='Literal vector dot products, undefined zero-vector cosine, and analytic derivatives at w=2 of 2w², 3w, w².',
        gradient_parts=parts,joint_gradient=joint,source_sha256=sha(__file__),measure_source_sha256=sha(Path(__file__).with_name('measure.py'))))
    print('Passed literal geometry and analytic chain-rule oracle',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);check(p.parse_args().output.resolve())
