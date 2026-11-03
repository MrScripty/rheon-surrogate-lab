"""Audit complete frozen GPU fusion evidence and compare matched timing scopes."""
import argparse
import collections
import json
import hashlib
import statistics
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FIELDS=('equation','pressure','velocity','tracer')
def read(path):return json.loads((ROOT/path).read_text())
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def key(row):return (row['identifier'],row['K'],row.get('start'),row['step'])
def run(args):
    if args.output.exists():raise ValueError('Fresh versioned report required')
    baseline_path='runs/full-ai-data-diversity-v1/expanded-generator-blocks-02/results.json'
    legacy_path='runs/full-ai-data-diversity-v1/expanded-legacy-rollout-02/results.json'
    baseline=read(baseline_path);legacy=read(legacy_path)
    original={key(r):r['output_sha256'] for r in baseline['rows']}
    original_legacy={key(r):r['output_sha256'] for r in legacy['rows']}
    bt={(r['identifier'],r['K'],r['start']):r for r in baseline['timings']}
    variants=[];sources={};evidence={baseline_path:digest(baseline_path),legacy_path:digest(legacy_path)}
    for number in args.variants:
        paths={kind:'runs/full-ai-data-diversity-v1/fused-'+kind+'-'+str(number).zfill(2)+'/results.json' for kind in ('generator-blocks','legacy-rollout')}
        block=read(paths['generator-blocks']);rollout=read(paths['legacy-rollout'])
        for kind,data,expected in [('blocks',block,original),('legacy',rollout,original_legacy)]:
            if (not data['status'].startswith('completed_') or not data['all_accuracy_pass'] or not data['every_output_bitwise_equal_corrected_baseline']
                    or len(data['rows'])!=len(expected) or {key(r):r['output_sha256'] for r in data['rows']}!=expected):
                raise ValueError('Incomplete complete-output accuracy/bitwise evidence: '+kind)
            if data['cpu_model_forward'] or data['optimizer_updates']!=0 or data['expanded_reserved52_references_opened']!=0:
                raise ValueError('Execution policy violated')
            for path,want in data['source_sha256'].items():
                if path in sources and sources[path]!=want:raise ValueError('Conflicting frozen source binding: '+path)
                sources[path]=want
        for path in paths.values():evidence[path]=digest(path)
        grouped=collections.defaultdict(list)
        for row in block['timings']:grouped[(row['cells_per_metre'],tuple(row['counts_xyz']),row['K'],row['internal_substeps'])].append(row)
        timings=[]
        for (density,shape,k,n),rows in sorted(grouped.items()):
            scopes={}
            for scope in rows[0]['amortized_ms_per_published_step']:
                medians=[r['amortized_ms_per_published_step'][scope]['median'] for r in rows]
                maximum=max(r['amortized_ms_per_published_step'][scope]['maximum'] for r in rows)
                ratios=[bt[(r['identifier'],r['K'],r['start'])]['amortized_ms_per_published_step'][scope]['median']/r['amortized_ms_per_published_step'][scope]['median'] for r in rows if scope in bt[(r['identifier'],r['K'],r['start'])]['amortized_ms_per_published_step']]
                scopes[scope]=dict(median_ms_per_step_range=[min(medians),max(medians)],observed_maximum_ms_per_step=maximum,
                    median_of_matched_baseline_speed_ratios=statistics.median(ratios) if ratios else None,full_block_observed_maximum_ms=maximum*k)
            timings.append(dict(cells_per_metre=density,counts_xyz=list(shape),K=k,internal_substeps=n,timing=scopes))
        variants.append(dict(number=number,fusion_mode=block['model_identity']['fusion_mode'],known_controls_on_gpu='known_control_identity' in block,known_control_build=block.get('known_control_identity',{}).get('build'),transfer_adapter=block.get('transfer_adapter'),device=block['device'],
            expanded_scenes=24,expanded_complete_outputs=len(block['rows']),consumed_regression_scenes=16,
            consumed_240step_complete_outputs=len(rollout['rows']),every_output_bitwise_equal_corrected_baseline=True,
            all_accuracy_pass=True,worst_expanded_errors={f:max(r['metrics']['relative'][f] for r in block['rows']) for f in FIELDS},
            worst_legacy_errors={f:max(r['metrics']['relative'][f] for r in rollout['rows']) for f in FIELDS},timings=timings,
            evidence=paths))
    for path,want in sources.items():
        if digest(path)!=want:raise ValueError('Frozen source or scientific evidence changed: '+path)
    evidence[str(Path(__file__).resolve().relative_to(ROOT))]=digest(str(Path(__file__).resolve().relative_to(ROOT)))
    report=dict(status='verified_gpu_execution_fusion_progress',created_unix=time.time(),variants=variants,
        frozen_bindings_verified=len(sources),evidence_sha256=evidence,optimizer_updates=0,cpu_model_forward=False,
        learned_weights_and_precisions_unchanged=True,known_controls_gpu_exp_rounding_separately_qualified=True,new_independent_confirmation=False,expanded_reserved52_references_opened=0,
        full_expanded_240step_trajectories_verified=False,above100axis_full_state_verified=False,
        matched_aspect_voxel_scaling_verified=False,historical_cpu_latency_target_verified=False,
        full_generator_targets_verified={str(k):False for k in range(1,5)},long600step_phase_started=False,
        timing_scope='GPU-only complete K-output generator, all explicit internal substeps and independent output ownership. Provided-control resident/CUDA-event scope excludes known-control generation. Variant3 adds resident-state plus complete known-control construction/upload/GPU generation; full host-input/output scope includes state upload and every intermediate download. Setup/capture/warm-up separate.',
        limits='Consumed development evidence only; three repetitions per short block are characterization, not the ordered maximum-latency/independent-confirmation gate. No CPU performance claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Fusion report written:',args.output,'verified frozen bindings:',len(sources))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',type=int,nargs='+',required=True);run(p.parse_args())
