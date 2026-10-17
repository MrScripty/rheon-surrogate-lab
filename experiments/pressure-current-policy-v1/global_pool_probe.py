"""Check whether the existing global-mean branch survives pressure gauge removal.

CPU-only analytical preparation for a subsequent architecture proposal. It never
changes a saved checkpoint, trains, simulates, selects or promotes a candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'water-residual-learning-v1'))
import audit


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','project','artifacts','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--core',type=int,required=True);args=parser.parse_args();root=args.root.resolve(strict=True)
    project=args.project.resolve(strict=True);artifacts=args.artifacts.resolve(strict=True);output=args.output.resolve()
    if output.exists()or not all(p.is_relative_to(root)for p in (project,artifacts,output)):
        raise ValueError('root-contained captured model and fresh output required')
    started=time.perf_counter();audit.initialize(project,artifacts,[args.core]);torch=audit.TORCH;network=audit.NETWORK
    original=network.global_pool
    class ZeroPool(torch.nn.Module):
        def forward(self,values):return torch.zeros_like(values[:,:,:1,:1,:1])
    rows=[]
    with torch.no_grad():
        for counts in ((16,16,16),(33,11,9),(40,40,40)):
            axes=[torch.arange(n,dtype=torch.float32)+.5 for n in counts[::-1]]
            z,y,x=torch.meshgrid(*axes,indexing='ij')
            low=torch.cos(torch.pi*x/counts[0]);mixed=low+.25*torch.cos(5*torch.pi*y/counts[1])
            for kind,field in (('pure-low',low),('low-plus-high',mixed)):
                values=(field-field.mean())[None,None].contiguous()
                network.global_pool=original;full=network(values)
                try:
                    network.global_pool=ZeroPool();without=network(values)
                finally:network.global_pool=original
                # f64 subtraction diagnoses rounding of f32 outputs; neither
                # network nor model inputs run above the deployed f32 precision.
                delta=full.to(torch.float64)-without.to(torch.float64)
                gauge_full=full.to(torch.float64)-full[0,0,0,0,0].item()
                gauge_delta=delta-delta[0,0,0,0,0]
                denominator=float(torch.linalg.vector_norm(gauge_full))
                rows.append({'counts_xyz':list(counts),'kind':kind,'network_calls':2,
                    'raw_branch_mean_effect':float(delta.mean()),
                    'raw_branch_effect_spread':float(delta.max()-delta.min()),
                    'gauge_output_relative_difference':float(torch.linalg.vector_norm(gauge_delta))/denominator,
                    'gauge_output_max_absolute_difference':float(gauge_delta.abs().max()),
                    'finite':bool(torch.isfinite(full).all()and torch.isfinite(without).all())})
    if not all(row['finite']for row in rows)or torch.cuda.is_initialized():raise ValueError('finite CPU diagnostic required')
    result={'format':'pressure-global-mean-gauge-probe-v1','status':'completed','source_sha256':digest(Path(__file__)),
        'audit_sha256':digest(Path(audit.__file__)),'source_witness_sha256':digest(project/'source-witness.json'),
        'weights_sha256':digest(artifacts/'weights.npz'),'network_dtype':'float32','metric_accumulation_dtype':'float64',
        'rows':rows,'network_calls':sum(r['network_calls']for r in rows),'cpu_threads':torch.get_num_threads(),
        'cuda_initialized':torch.cuda.is_initialized(),'optimizer_updates':0,'simulations_advanced':0,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_s':time.perf_counter()-started,
        'scope':'Analytical fields on already consumed grid shapes; in-memory removal of the spatially constant global mean branch, restored after each probe. No saved weight or inference package change. Relative differences include float32 rounding; this is not an accuracy or latency qualification.'}
    with output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
