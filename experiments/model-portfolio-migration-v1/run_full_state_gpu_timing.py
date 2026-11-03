"""Register and own one balanced frozen full-state GPU timing campaign.

Only original metadata/current-state archives and frozen executables are read.
Each process owns one execution, and failure stops the original campaign without
retry. Diagnostic-only child groups are interruption-safe: original scientific
files and live application state are never written by these workers.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from full_state_gpu_timing import ROOT, EXECUTIONS, read, save, sha, resources

BASE = ROOT/'runs/model-portfolio-migration-v1'
ORIGINALS = dict(cell='docs/plans/full-ai-data-diversity-v1/cell-coordinate-benchmark-protocol-01.json',
    temporal='docs/plans/full-ai-data-diversity-v1/temporal-frozen-GPU-benchmark-protocol-02.json')


def prepare(path, output):
    if (path.exists() or output.exists() or not path.is_relative_to(BASE)
            or not output.is_relative_to(BASE)):
        raise ValueError('Fresh task-owned protocol and campaign root required')
    sources = {}; original_sha = {}
    for family, name in ORIGINALS.items():
        original_sha[family] = sha(ROOT/name)
        original = read(ROOT/name)
        for relative, expected in original['source_sha256'].items():
            if relative in sources and sources[relative] != expected:
                raise ValueError('Original protocol dependency identities disagree: '+relative)
            if sha(ROOT/relative) != expected:
                raise ValueError('Original frozen execution source differs: '+relative)
            sources[relative] = expected
        sources[name] = original_sha[family]
        if family == 'cell':
            references = [original[key] for key in ('bank','baseline_outputs','candidate_outputs',
                          'candidate_full_history','legacy_qualification')]
            references += [row[key] for row in original['window_sources'] for key in ('controller','integrity')]
            if not all(read(ROOT/original[key])['all_accuracy_pass']
                       for key in ('baseline_outputs','candidate_outputs')):
                raise ValueError('The original complete cell-coordinate/bound populations must be accuracy-qualified')
        else:
            references = [original['qualification'],*original['audits']]
            qualified=read(ROOT/original['qualification'])
            if not (qualified['all_outputs_bitwise_equal_eager'] and qualified['fine_baseline_all_accuracy_pass']):
                raise ValueError('Original eager/graph equivalence and fine baseline qualification required')
        for relative in references:
            expected=sha(ROOT/relative)
            if relative in sources and sources[relative]!=expected:
                raise ValueError('Original accuracy/workload evidence changed: '+relative)
            sources[relative]=expected
    for name in ('full_state_gpu_timing.py','run_full_state_gpu_timing.py','test_full_state_gpu_timing.py'):
        relative='experiments/model-portfolio-migration-v1/'+name
        sources[relative]=sha(ROOT/relative)
    runtime='runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    sources[runtime]=sha(ROOT/runtime)
    environment='experiments/pressure-four-pass-distillation-v1/common.py'
    sources[environment]=sha(ROOT/environment)
    browser='runs/model-portfolio-migration-v1/history-cache-progression-browser-01/report.json'
    report=read(ROOT/browser)
    if report['status'] != 'passed_native_project_UI_actions' or not report['browser_stopped']:
        raise ValueError('The original UI campaign must be terminal before timing')
    observation=resources()
    if observation['other_compute_pids']:
        raise RuntimeError('Required GPU compute environment unavailable')
    order=list(EXECUTIONS)
    blocks=[dict(execution=name,phase=phase) for phase,names in ((0,order),(1,list(reversed(order))))
            for name in names]
    protocol=dict(schema='rheon.frozen-complete-full-state-GPU-timing-protocol.v1',
        created_unix=time.time(),output_root=str(output.relative_to(ROOT)),
        worker='experiments/model-portfolio-migration-v1/full_state_gpu_timing.py',
        runtime_manifest=runtime,source_sha256=sources,original_protocols=ORIGINALS,
        original_protocol_sha256=original_sha,host_cpu=4,torch_threads=1,interop_threads=1,
        block_order=blocks,samples_per_case_per_block=10,warmups_per_case=5,
        expected_cases=dict(cell=384,temporal=216),expected_timed_calls=24000,
        expected_timed_physical_outputs=60000,expected_first_and_warmup_physical_outputs=36000,
        worker_deadline_seconds=1800,aggregate_deadline_seconds=7200,
        maximum_Torch_GPU_allocated_bytes=4*1024**3,maximum_worker_RSS_bytes=8*1024**3,
        target_ms_per_physical_output=0.2,target_scope='Report all30/60/120cadences andK1..4 individually; target does not change with display cadence. No mixed-population Pareto or hard real-time claim.',
        timing_scope='Complete owned CPU NumPy pressure/vx/vy/vz/tracer buffers and prescribed scene/time/K controls through common dtype/extent/contiguity validation and NumPy/Torch views, captured packing/H2D, actual known-control evaluation and every original learned advance, reconstruction/allocation/D2H/synchronization, and common flat NumPy output views for allKowned physical states. CPU and GPU work inside this complete wall interval. Divide byKphysical outputs only. Model/file/graph/geometry setup, first calls, warmups, source hashing, finite-input qualification, output/finiteness/ownership oracles and resource sampling separate. No RHS/future fields passed to a generator.',
        case_order='Canonical scene/start/cadence/K/schedule order in phase0, reversed in phase1. Four process executions followed by their reversed order; pool20samples per exact case/execution.',
        numerical_contract='Original source, parameters, output hashes and physical integration schedules. Original pressure/equation/drift definitions remain distinct. Accuracy is inherited only after all timed/warm output field bytes match their exact frozen case.',
        readiness=dict(browser_report=browser,browser_report_sha256=sha(ROOT/browser),resource_observation=observation),
        CPU_model_forwards=False,new_parameter_fits=0,new_reference_generation=False,
        reference_solver_epoch='frozen_old_Rheon',reserved_reference_sets_opened=0,
        unchanged_original_workers_replayed=False,previous_legacy_latency_pass_flags_reused=False,
        variability_policy='Retain every sample, per-case median/p95/maximum and all observed maxima. Sampled contention evidence is not uninterrupted hardware exclusivity.',
        resource_budget_authority='Existing temporal qualification4GiBTorchallocation limit;8GiBhost worker cap fits current62GiBmachine alongside open UI. Finite1800sworker/7200scampaign ownership; these are experiment bounds, not application latency targets.')
    save(path,protocol)
    print('Registered',len(sources),'source bindings,8process blocks,24000timed calls/60000physical outputs',flush=True)


def run(path):
    protocol=read(path);output=ROOT/protocol['output_root']
    if output.exists() or not output.is_relative_to(BASE):
        raise ValueError('Observe the original campaign; no replay into an existing root')
    for name,expected in protocol['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Registered source differs: '+name)
    runtime=read(ROOT/protocol['runtime_manifest'])
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir(parents=True)
    state=dict(status='starting',pid=os.getpid(),created_unix=time.time(),protocol_sha256=sha(path),
        completed=[],active=None,inference_device='cuda',CPU_model_forwards=0,
        full_portfolio_complete=False,source_or_dataset_copies=0)
    begun=time.monotonic()
    try:
        for index,block in enumerate(protocol['block_order']):
            remaining=protocol['aggregate_deadline_seconds']-(time.monotonic()-begun)
            if remaining<=0:
                raise TimeoutError('Registered original campaign deadline')
            name=str(index+1).zfill(2)+'-'+block['execution']
            destination=output/name
            command=[runtime['python'],'-B',str(ROOT/protocol['worker']),'--protocol',str(path),
                     '--execution',block['execution'],'--phase',str(block['phase']),'--output',str(destination)]
            with (output/(name+'.log')).open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=environment(runtime,'cuda'),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='original_full_state_GPU_worker_running',
                    active=dict(block=index+1,pid=process.pid,output=str(destination.relative_to(ROOT)),**block))
                save(output/'status.json',state)
                try:
                    returncode=process.wait(timeout=min(remaining,protocol['worker_deadline_seconds']+60))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            result_path=destination/'results.json'
            result=read(result_path) if result_path.exists() else None
            if (returncode or result is None or result['status']!='complete_original_full_state_GPU_timing'
                    or not result['cleanup_complete'] or result['device_qualification']['forbidden_module_calls']
                    or len(result['rows'])!=protocol['expected_cases'][EXECUTIONS[block['execution']][0]]):
                raise ValueError('Original full-state worker incomplete; retain it and reconcile without retry')
            state['completed'].append(dict(block=index+1,**block,result=str(result_path.relative_to(ROOT)),
                                           sha256=sha(result_path),cases=len(result['rows'])))
            state.update(status='original_full_state_GPU_block_complete',active=None)
            save(output/'status.json',state);print(name,'complete',len(result['rows']),'cases',flush=True)
        for name,expected in protocol['source_sha256'].items():
            if sha(ROOT/name)!=expected:
                raise ValueError('Source changed during original campaign: '+name)
        state.update(status='complete_original_frozen_full_state_GPU_timing',active=None,
                     finished_unix=time.time(),independent_reduction_pending=True,source_bindings_unchanged=True)
    except BaseException as error:
        state.update(status='original_full_state_GPU_campaign_requires_reconciliation',
                     error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(output/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    register=commands.add_parser('prepare');register.add_argument('--protocol',type=Path,required=True)
    register.add_argument('--output',type=Path,required=True)
    execute=commands.add_parser('run');execute.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.protocol.resolve(),args.output.resolve())
    else:
        run(args.protocol.resolve())
