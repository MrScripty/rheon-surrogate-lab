"""Measure full train-only physical increments and prescribed-force scales.

This prepares a normalization experiment, not a model or a target change. Labels
are used only for aggregate training statistics, never query-time normalization.
"""
import argparse
import os
from pathlib import Path
import math
import torch
from temporal_data import load,make_features
from study_support import HERE,BASE,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh training-only step statistics required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',
                        [view/row['path']for row in manifest['shards']],'cpu')
    totals={k:{field:dict(squared=0.,elements=0)for field in ('velocity','tracer')}for k in (1,2,3,4)}
    force_total=dict(squared=0.,elements=0);rows=[]
    with torch.inference_mode():
        for parent in parents:
            counts=parent['scene']['counts'];nx,ny,nz=counts;states=parent['states']
            masks=((slice(0,nz),slice(0,ny),slice(0,nx+1)),
                   (slice(0,nz),slice(0,ny+1),slice(0,nx)),
                   (slice(0,nz+1),slice(0,ny),slice(0,nx)))
            row=dict(identifier=parent['identifier'],counts_xyz=counts,states=len(states),increments={})
            for k in (1,2,3,4):
                local={field:dict(squared=0.,elements=0)for field in ('velocity','tracer')}
                for component,selection in enumerate(masks,1):
                    value=states[:,component][(slice(None),*selection)].double()
                    delta=value[k:]-value[:-k]
                    local['velocity']['squared']+=float(delta.square().sum());local['velocity']['elements']+=delta.numel()
                value=states[:,4,:nz,:ny,:nx].double();delta=value[k:]-value[:-k]
                local['tracer']['squared']=float(delta.square().sum());local['tracer']['elements']=delta.numel()
                row['increments'][str(k)]={field:math.sqrt(spec['squared']/spec['elements'])for field,spec in local.items()}
                for field,spec in local.items():
                    for key,value in spec.items():totals[k][field][key]+=value
            forces=make_features(parent,norm['scales'],1,'cpu')['forces'][:128]
            for component,selection in enumerate(masks):
                value=forces[:,component][(slice(None),*selection)].double()
                force_total['squared']+=float(value.square().sum());force_total['elements']+=value.numel()
            rows.append(row)
    normalized={str(k):{field:math.sqrt(spec['squared']/spec['elements'])for field,spec in groups.items()}for k,groups in totals.items()}
    force_scale=math.sqrt(force_total['squared']/force_total['elements'])
    if not 0<force_scale<1 or any(not 0<value<1 for groups in normalized.values()for value in groups.values()):
        raise ValueError('Unexpected step statistics; inspect rather than silently clamp')
    write(args.output,dict(schema='rheon-temporal-step-scale-statistics-v1',status='verified_train_only_statistics',
        normalized_increment_rms=normalized,normalized_prescribed_force_rms=force_scale,
        physical_increment_rms={str(k):{field:value*norm['scales'][field]for field,value in groups.items()}for k,groups in normalized.items()},
        aggregate_sums={str(k):value for k,value in totals.items()},known_force_aggregate=force_total,parents=rows,
        fitting_view_sha256=sha(view/'manifest.json'),original_normalization_sha256=sha(view/'normalization.json'),
        source_sha256=sha(__file__),training_parent_count=len(parents),development_or_reserved_used=False,
        original_field_error_normalization_unchanged=True,optimizer_updates=0,candidate_accuracy_claim=False,
        usage='Fixed aggregate training scales for a separately registered model conditioning follow-up; never future query labels'))
    print('Prepared train-only per-horizon step scales',normalized,'forcing',force_scale,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
