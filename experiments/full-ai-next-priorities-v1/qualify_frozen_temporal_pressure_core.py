"""Compare the differentiable learned pressure port with its unchanged parent.

Native future requests are offline qualification inputs only. This script cannot
qualify a causal temporal model or produce temporal speed/accuracy frontier data.
"""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from frozen_temporal_pressure_core import FrozenPressureCore,index_laplacian
from precise_routed_tail_pressure import PreciseRoutedTailPrediction
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha,load_checkpoint,equation_error


def relative(a,b):return float((a-b).norm())/max(float(b.norm()),1e-30)


def run(args):
    if args.output.exists():raise ValueError('Fresh component qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    registration=read(DOCS/'temporal-pressure-core-port-registration-02.json')
    if registration['source_sha256']!=sha(HERE/'frozen_temporal_pressure_core.py'):
        raise ValueError('Port changed after qualification registration')
    assets=BASE/'temporal-pressure-core-assets-01'
    if sha(assets/'receipt.json')!=registration['assets_receipt_sha256']:raise ValueError('Frozen asset identity changed')
    # The immutable legacy loader intentionally requires a CPU-only context at
    # load time. Restore its CPU reference before this qualification initializes
    # CUDA for the new differentiable component.
    state=load_checkpoint('a-tail8')
    predict_parent=PreciseRoutedTailPrediction(state)
    core=FrozenPressureCore(assets).eval().to(args.device)
    cpu=FrozenPressureCore(assets).eval()
    bank=read(BASE/'temporal-data-01/bank.json')
    parents={}
    for row in bank['trajectories']:
        if row['split']!='train':continue
        receipt=read(ROOT/row['receipt']);scene=receipt['scene']
        key=(tuple(scene['counts']),scene['cells_per_metre'])
        parents.setdefault(key,(row,receipt))
    if len(parents)!=8:raise ValueError('Four grids and both spatial densities required')
    rows=[];gradients=[];stencils=[]
    before={name:value.detach().cpu().numpy().tobytes()for name,value in core.state_dict().items()}
    generator=torch.Generator().manual_seed(2026101923)
    for (counts,cpm),(row,receipt) in parents.items():
        shape=(1,1,*counts[::-1]);h=torch.tensor([1/cpm],dtype=torch.float64)
        rho=torch.tensor([1000.],dtype=torch.float64)
        probe=torch.randn(shape,generator=generator,dtype=torch.float64)
        expected=state.model.owner.full_operator(probe,h,rho)*(rho/h).reshape(-1,1,1,1,1)
        stencil_difference=relative(index_laplacian(probe),expected)
        if stencil_difference>1e-14:raise ValueError('Dimensionless forward stencil differs from original operator')
        stencils.append(dict(counts_xyz=counts,cells_per_metre=cpm,relative_difference=stencil_difference))
        for start in (0,8,63,120):
            requests=[];samples=[];references=[]
            for horizon in (1,2,3,4):
                record=receipt['states'][start+horizon]
                path=(ROOT/row['receipt']).parent/record['path']
                if sha(path)!=record['sha256']:raise ValueError('Original native request changed')
                with np.load(path,allow_pickle=False)as arrays:
                    rhs=torch.from_numpy(arrays['rhs'].copy().reshape(shape))
                    reference=torch.from_numpy(arrays['pressure'].copy().reshape(shape))
                samples.append(dict(base=torch.zeros_like(rhs),native_input_residual=rhs,rhs=rhs,
                    cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64)))
                requests.append(rhs*(rho/h).reshape(-1,1,1,1,1));references.append(reference)
            q=torch.cat(requests)
            with torch.no_grad():
                output=core(q.to(args.device)).cpu();cpu_output=cpu(q)
                for index,sample in enumerate(samples):
                    parent=predict_parent(sample)
                    predicted=output[index:index+1]
                    difference=relative(predicted-predicted.mean(),parent-parent.mean())
                    device_difference=relative(predicted-predicted.mean(),cpu_output[index:index+1]-cpu_output[index:index+1].mean())
                    equation=equation_error(state,predicted,sample);parent_equation=equation_error(state,parent,sample)
                    reference=references[index]-references[index].mean()
                    field=relative(predicted-predicted.mean(),reference)
                    parent_field=relative(parent-parent.mean(),reference)
                    if difference>registration['pressure_relative_difference_tolerance']:
                        raise ValueError(f'Parent prediction parity exceeded tolerance: {difference}')
                    if device_difference>registration['CPU_device_pressure_relative_tolerance']:
                        raise ValueError(f'CPU/device pressure parity exceeded tolerance: {device_difference}')
                    if abs(equation-parent_equation)>max(registration['equation_absolute_tolerance'],
                            registration['equation_relative_tolerance']*parent_equation):
                        raise ValueError(f'Parent equation parity exceeded tolerance: {equation}, {parent_equation}')
                    if (max(equation,field)<=.004)!=(max(parent_equation,parent_field)<=.004):
                        raise ValueError('Port changes pressure/equation oracle classification')
                    if not torch.isfinite(predicted).all()or float(predicted[0,0,0,0,0])!=0:
                        raise ValueError('Finite cell-zero pressure gauge required')
                    individual=core(q[index:index+1].to(args.device)).cpu()
                    batch_difference=relative(predicted,individual)
                    if batch_difference>registration['pressure_relative_difference_tolerance']:
                        raise ValueError('Batch prediction changes individual result')
                    rows.append(dict(identifier=row['identifier'],counts_xyz=counts,cells_per_metre=cpm,start=start,
                        horizon=index+1,parent_relative_difference=difference,CPU_device_relative_difference=device_difference,
                        batch_individual_relative_difference=batch_difference,pressure_relative_l2=field,
                        equation_relative_residual=equation,parent_pressure_relative_l2=parent_field,
                        parent_equation_relative_residual=parent_equation,gauge_exact=True))
            if start==8:
                for batch in (1,2,3,4):
                    input_q=q[:batch].to(args.device).clone().requires_grad_();previous=None
                    for repeat in range(2):
                        value=core(input_q);objective=value.square().mean()
                        gradient,=torch.autograd.grad(objective,input_q)
                        if not bool(torch.isfinite(gradient).all())or float(gradient.norm())<=0:
                            raise ValueError('Active finite input gradients required')
                        if previous is not None and not torch.equal(previous,gradient):
                            raise ValueError('Repeated component gradients differ')
                        previous=gradient.detach().clone()
                        euler=abs(float((gradient*input_q).sum().detach()-2*objective.detach()))/max(float(2*objective.detach()),1e-30)
                        if euler>registration['homogeneous_gradient_relative_tolerance']:
                            raise ValueError(f'Gradient fails local homogeneity: {euler}')
                    gradients.append(dict(counts_xyz=counts,cells_per_metre=cpm,batch=batch,
                        gradient_norm=float(gradient.norm()),repeated_gradient_bitwise_equal=True,
                        homogeneous_gradient_relative_error=euler,optimizer_updates=0))
        with torch.no_grad():
            zero=core(torch.zeros(shape,dtype=torch.float64,device=args.device))
            if torch.count_nonzero(zero):raise ValueError('Zero request must produce exact zero pressure')
        print(f'Qualified {counts}, {cpm} cells/metre, parent parity and batched gradients',flush=True)
    if any(value.detach().cpu().numpy().tobytes()!=before[name]for name,value in core.state_dict().items()):
        raise ValueError('Frozen coefficients changed during qualification')
    rejected=False
    try:core(torch.zeros((1,1,7,7,7),device=args.device))
    except ValueError:rejected=True
    if not rejected:raise ValueError('Unsupported geometry must reject')
    write(args.output,dict(status='passed',device=args.device,rows=rows,gradients=gradients,stencils=stencils,
        registration_sha256=sha(DOCS/'temporal-pressure-core-port-registration-02.json'),
        source_sha256={name:sha(HERE/name)for name in('frozen_temporal_pressure_core.py','export_temporal_pressure_assets.py',
            'qualify_frozen_temporal_pressure_core.py')},assets_receipt_sha256=sha(assets/'receipt.json'),
        optimizer_updates=0,parameter_values_unchanged=True,unsupported_geometry_rejected=True,
        frozen_spatial_parameters=sum(p.numel()for p in core.parameters()),trainable_parameters=sum(p.numel()for p in core.parameters()if p.requires_grad),
        dense_learned_map_preparation=core.dense_preparation,parent_execution='Independently confirmed precise routed learned pressure implementation',
        full_state_temporal_candidate=False,causal_prediction_accuracy_claim=False,CPU_latency_claim=False,
        native_future_RHS_used_only_for_offline_component_qualification=True,reserved_scenes_consumed=False,
        candidate_numerical_pressure_solve=False,analytical_inverse_coefficients_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if not args.output.exists():write(args.output,dict(status='failed',device=args.device,error=type(error).__name__+': '+str(error),
            qualification_source_sha256=sha(__file__),optimizer_updates=0,full_state_temporal_candidate=False))
        raise
