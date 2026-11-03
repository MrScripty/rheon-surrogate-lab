"""Offline arbitrary-field parity and residual-operator stability diagnostics.

Power and singular-gain probes are observations, not universal stability proofs.
No numerical solve or correction is added to candidate inference.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from full_learned_pressure_map import FullLearnedPressureMap
from frozen_temporal_pressure_core import index_laplacian
from study_support import HERE,BASE,DOCS,read,write,sha


def center(value):return value-value.mean(dim=(-3,-2,-1),keepdim=True)
def normalize(value):return value/value.flatten(1).norm(dim=1).reshape(-1,1,1,1,1).clamp_min(1e-30)
def relative(a,b):return float((a-b).norm())/max(float(b.norm()),1e-30)


def run(args):
    if args.output.exists():raise ValueError('Fresh arbitrary-field/operator audit required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=read(DOCS/'full-learned-pressure-map-audit-registration-01.json')
    if reg['module_sha256']!=sha(HERE/'full_learned_pressure_map.py'):raise ValueError('Registered neural map changed')
    core=FullLearnedPressureMap(BASE/'temporal-pressure-core-assets-01',BASE/'learned-pressure-gate-bounds-01.json').eval()
    generator=torch.Generator().manual_seed(2026102001);rows=[];parity=[]
    with torch.no_grad():
        for counts,route in core.routes.items():
            if route is None:continue
            shape=(4,1,*counts)
            raw=torch.randn(shape,dtype=torch.float64,generator=generator)
            checker=sum(torch.meshgrid(*[torch.arange(n)for n in counts],indexing='ij'))%2
            raw[0,0]=checker.double()*2-1;raw[1]=index_laplacian(raw[1:2])[0]
            raw[2].zero_();raw[2,0,0,0,0]=1.
            raw=center(raw)
            for scale in (1.,1e-10,1e10):
                q=raw*scale;actual=core(q);parent=core.source(q)
                difference=relative(center(actual),center(parent))
                residual=(index_laplacian(actual)-q).flatten(1).norm(dim=1)/q.flatten(1).norm(dim=1)
                old_residual=(index_laplacian(parent)-q).flatten(1).norm(dim=1)/q.flatten(1).norm(dim=1)
                if difference>reg['pressure_relative_tolerance']:raise ValueError(f'Arbitrary-field neural map parity exceeded: {difference}')
                if not torch.equal(residual<=.004,old_residual<=.004):raise ValueError('Arbitrary-field threshold classification changed')
                parity.append(dict(counts_zyx=counts,scale=scale,pressure_relative_difference=difference,
                    residual_errors=residual.tolist(),parent_residual_errors=old_residual.tolist()))
            matrix=getattr(core,route)
            def residual(value):return center(value-index_laplacian(core(center(value))))
            def transpose(value):
                value=center(value)
                return center(value-torch.nn.functional.linear(index_laplacian(value).flatten(1),matrix.T).reshape_as(value))
            x=center(torch.randn(shape,dtype=torch.float64,generator=generator));y=center(torch.randn(shape,dtype=torch.float64,generator=generator))
            a=float((residual(x)*y).sum());b=float((x*transpose(y)).sum())
            adjoint_error=abs(a-b)/max(abs(a),abs(b),1e-30)
            if adjoint_error>reg['adjoint_relative_tolerance']:raise ValueError('Residual adjoint diagnostic failed')
            value=normalize(x);power=[]
            for iteration in range(reg['power_iterations']):
                result=residual(value);gain=result.flatten(1).norm(dim=1);power.append(gain.tolist());value=normalize(result)
            value=normalize(y);singular=[]
            for iteration in range(reg['power_iterations']):
                result=residual(value);singular.append(result.flatten(1).norm(dim=1).tolist());value=normalize(transpose(result))
            last=np.asarray(power[-8:]);growing=bool((last.min(axis=0)>reg['observed_growth_threshold']).any())
            rows.append(dict(counts_zyx=counts,power_gains=power,singular_gain_probes=singular,
                last_eight_power_gain_minimum_by_probe=last.min(axis=0).tolist(),
                last_eight_power_gain_maximum_by_probe=last.max(axis=0).tolist(),
                measured_singular_gain_lower_bound=max(singular[-1]),observed_persistently_growing_mode=growing,
                residual_adjoint_relative_difference=adjoint_error,universal_stability_proof=False))
            print('Audited learned residual operator',counts,'persistent growth',growing,'last gains',power[-1],flush=True)
    write(args.output,dict(status='completed',rows=rows,arbitrary_field_parity=parity,
        observed_growing_residual_modes=any(row['observed_persistently_growing_mode']for row in rows),
        registration_sha256=sha(DOCS/'full-learned-pressure-map-audit-registration-01.json'),
        source_sha256={n:sha(HERE/n)for n in('audit_full_learned_map.py','full_learned_pressure_map.py','frozen_temporal_pressure_core.py')},
        optimizer_updates=0,analytical_inverse_constructed=False,numerical_solver_at_candidate_inference=False,
        candidate_frontier_point=False,CPU_latency_claim=False,reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if not args.output.exists():write(args.output,dict(status='failed',error=type(error).__name__+': '+str(error),source_sha256=sha(__file__),optimizer_updates=0))
        raise
