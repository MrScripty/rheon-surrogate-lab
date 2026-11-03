"""Recompute every saved offline pressure error with an independent NumPy stencil."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action(pressure,h):
    value=np.zeros_like(pressure)
    for axis in (1,2,3):
        lower=[slice(None)]*4;upper=lower.copy()
        lower[axis]=slice(None,-1);upper[axis]=slice(1,None)
        difference=pressure[tuple(lower)]-pressure[tuple(upper)]
        value[tuple(lower)]+=difference;value[tuple(upper)]-=difference
    return value*(h/1000.)


def norm(value):return np.sqrt(np.square(value).sum(axis=(1,2,3)))


def run(args):
    if args.output.exists():raise ValueError('Fresh independent output audit required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Audit source/evidence changed: '+path)
    directory=ROOT/reg['evaluation'];result=read(directory/'results.json')
    if result['causal_full_state_generator'] or result['own_history_rollout'] or result['candidate_frontier_point'] or not result['teacher_rhs_used']:
        raise ValueError('Offline component scope must remain explicit')
    bank={row['identifier']:row for row in read(ROOT/reg['bank'])['trajectories']}
    if len(bank)!=8 or {row['identifier']for row in result['pressure_outputs']}!=set(bank):raise ValueError('All eight consumed development scenes required')
    metric_rows={(row['identifier'],row['state']):row for row in result['rows']}
    if len(metric_rows)!=1920 or len(result['rows'])!=1920:raise ValueError('Complete unique 1920-state coverage required')
    verified=[];sources={}
    for output in result['pressure_outputs']:
        path=directory/output['path']
        if sha(path)!=output['sha256']:raise ValueError('Saved physical pressure changed')
        with np.load(path,allow_pickle=False)as archive:
            if archive.files!=['pressure']:raise ValueError('Complete pressure archive expected')
            pressure=archive['pressure'].copy()
        counts=output['counts_xyz'];shape=tuple(reversed(counts))
        if pressure.dtype.str!='<f8' or pressure.shape!=(240,*shape) or not np.isfinite(pressure).all() or np.count_nonzero(pressure[:,0,0,0]):
            raise ValueError('Finite complete double cell-zero physical pressure required')
        row=bank[output['identifier']];receipt_path=ROOT/row['receipt']
        if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Native scene receipt changed')
        receipt=read(receipt_path);truth=[];rhs=[]
        sources[str(path.relative_to(ROOT))]=sha(path)
        for step in range(1,241):
            spec=receipt['states'][step];native=receipt_path.parent/spec['path']
            if sha(native)!=spec['sha256']:raise ValueError('Native pressure/RHS archive changed')
            with np.load(native,allow_pickle=False)as archive:
                truth.append(archive['pressure'].reshape(shape));rhs.append(archive['rhs'].reshape(shape))
        truth=np.stack(truth).astype('<f8',copy=False);rhs=np.stack(rhs).astype('<f8',copy=False)
        residual=norm(action(pressure,1./receipt['scene']['cells_per_metre'])-rhs)/np.maximum(norm(rhs),1e-30)
        mean_axes=(1,2,3)
        centered=pressure-pressure.mean(axis=mean_axes,keepdims=True)
        truth=truth-truth.mean(axis=mean_axes,keepdims=True)
        error=norm(centered-truth)/np.maximum(norm(truth),1e-30)
        for step,(eq,p)in enumerate(zip(residual,error),1):
            reported=metric_rows[(output['identifier'],step)]
            for observed,field in ((eq,'equation_relative_l2'),(p,'mean_free_pressure_relative_l2')):
                if not np.isfinite(observed) or abs(float(observed)-reported[field])>reg['maximum_summary_absolute_difference']:
                    raise ValueError('Independent raw output differs from reported metrics')
            verified.append(dict(identifier=output['identifier'],state=step,equation_relative_l2=float(eq),mean_free_pressure_relative_l2=float(p),
                component_pass=bool(eq<=reg['maximum_error']and p<=reg['maximum_error'])))
        print(output['identifier'],'240 saved pressures independently verified',flush=True)
    audit=dict(status='independent_saved_pressure_audit_complete',physical_pressure_outputs_verified=len(verified),
        worst_equation_relative_l2=max(row['equation_relative_l2']for row in verified),
        worst_mean_free_pressure_relative_l2=max(row['mean_free_pressure_relative_l2']for row in verified),
        component_pressure_accuracy_verified=all(row['component_pass']for row in verified),rows=verified,
        source_evidence=sources,source_sha256=reg['source_sha256'],protocol_sha256=sha(args.protocol),
        independent_numpy_edge_flux_stencil=True,optimizer_updates=0,teacher_rhs_diagnostic_only=True,
        full_generator_targets_verified=False,own_history_rollout=False,fresh_holdout_consumed=False)
    args.output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
