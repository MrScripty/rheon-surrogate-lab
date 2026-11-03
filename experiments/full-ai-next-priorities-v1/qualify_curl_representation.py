"""Independent impulse, divergence, wall and small-grid rank checks of readout."""
import argparse
import math
import os
from pathlib import Path
import torch
from bundle_model_curl import velocity_curl,edge_masks
from temporal_features_v2 import geometry
from study_support import HERE,write,sha


def divergence(values):
    vx,vy,vz=values
    return vx[:,:,:,:,1:]-vx[:,:,:,:,:-1]+vy[:,:,:,1:,:]-vy[:,:,:,:-1,:]+vz[:,:,1:,:,:]-vz[:,:,:-1,:,:]


def free_faces(counts):
    nx,ny,nz=counts;_,mask=geometry(tuple(counts),'cpu')
    return tuple(mask[i,:s[0],:s[1],:s[2]].bool()for i,s in enumerate(((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx)),1))


def run(args):
    if args.output.exists():raise ValueError('Fresh curl representation qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.manual_seed(2026102001)
    potential=torch.zeros((1,1,3,4,4,4),dtype=torch.float64);potential[0,0,2,1,1,1]=1
    actual=velocity_curl(potential,(3,3,3));expected=tuple(torch.zeros_like(v)for v in actual)
    expected[0][0,0,1,0,1]=1;expected[0][0,0,1,1,1]=-1
    expected[1][0,0,1,1,0]=-1;expected[1][0,0,1,1,1]=1
    if not all(torch.equal(a,b)for a,b in zip(actual,expected)):raise ValueError('Independent four-face circulation impulse failed')
    rows=[]
    for counts in ((16,16,16),(9,11,11),(15,6,8),(32,32,32)):
        nx,ny,nz=counts;potential=torch.randn((2,4,3,nz+1,ny+1,nx+1),dtype=torch.float64,requires_grad=True)
        values=velocity_curl(potential,counts);div=divergence(values)
        scale=sum(value.square().sum()for value in values).sqrt()
        relative=float(div.detach().norm()/scale.detach())
        walls=(values[0][:,:,:,:,0],values[0][:,:,:,:,-1],values[1][:,:,:,0,:],values[1][:,:,:,-1,:],values[2][:,:,0,:,:],values[2][:,:,-1,:,:])
        if relative>1e-12 or any(bool((wall!=0).any())for wall in walls):raise ValueError('Divergence identity or exact normal wall condition failed')
        objective=sum(value.square().mean()for value in values);objective.backward()
        if potential.grad is None or not bool(torch.isfinite(potential.grad).all())or float(potential.grad.norm())==0:
            raise ValueError('Readout derivative disconnected')
        rows.append(dict(counts_xyz=list(counts),horizons=4,batch=2,divergence_relative_l2=relative,
            normal_wall_velocity_exactly_zero=True,finite_nonzero_backward=True))
    ranks=[]
    for counts in ((2,2,2),(3,3,3),(3,4,2)):
        nx,ny,nz=counts;edges=edge_masks(counts,'cpu').bool();positions=edges.nonzero();ne=len(positions)
        units=torch.zeros((ne,1,3,nz+1,ny+1,nx+1),dtype=torch.float64)
        units[torch.arange(ne),0,positions[:,0],positions[:,1],positions[:,2],positions[:,3]]=1
        values=velocity_curl(units,counts);face_masks=free_faces(counts)
        curl=torch.cat([value[:,0,mask]for value,mask in zip(values,face_masks)],dim=1).T
        nf=curl.shape[0];face_units=[];offset=0
        for value,mask in zip(values,face_masks):
            field=torch.zeros((nf,1,*value.shape[2:]),dtype=torch.float64);indices=mask.nonzero();n=len(indices)
            field[torch.arange(offset,offset+n),0,indices[:,0],indices[:,1],indices[:,2]]=1
            face_units.append(field);offset+=n
        div_matrix=divergence(face_units).flatten(1).T
        curl_rank=int(torch.linalg.matrix_rank(curl,atol=1e-10,rtol=0));div_rank=int(torch.linalg.matrix_rank(div_matrix,atol=1e-10,rtol=0))
        if curl_rank!=nf-div_rank or float((div_matrix@curl).abs().max())>1e-12:
            raise ValueError('Small-grid representation does not span the closed divergence-free face subspace')
        ranks.append(dict(counts_xyz=list(counts),interior_faces=nf,free_edge_potentials=ne,
            curl_rank=curl_rank,divergence_rank=div_rank,closed_divergence_free_dimension=nf-div_rank))
    write(args.output,dict(status='passed',rows=rows,small_grid_ranks=ranks,
        independent_four_face_impulse=True,source_sha256=sha(__file__),model_source_sha256=sha(HERE/'bundle_model_curl.py'),
        optimizer_updates=0,candidate_numerical_pressure_solve=False,candidate_velocity_post_projection=False,
        fixed_linear_curl_readout_inside_model=True,rank_tests_offline_only=True,
        caveat='Representational constraints do not establish prediction accuracy; float32 physical interchange has roundoff. Existing nonzero input divergence is not corrected.'))
    print('Qualified fixed curl readout: impulse, four-grid divergence/walls/backward and small-grid full subspace ranks',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
