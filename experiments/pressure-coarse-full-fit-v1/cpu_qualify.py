"""Sequential qualified CPU phases with one bounded concurrent scene campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from process_ownership import become_subreaper, cancel_and_drain, drain_adopted
from prefix import qualified_prefix

HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def interrupted(*_):raise KeyboardInterrupt('CPU phase owner cancelled')


def fixture():
    # Translate termination to the existing rollout owner's cleanup/finally path.
    sys.path.insert(0,str(HERE.parent/'pressure-throughput-feedback-v1'))
    import episode
    sys.argv[:]=[episode.__file__,*sys.argv[2:]]
    episode.main()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','project','study','runtime','runtime-owner','regression-template','timing-template','control-comparison','control-campaign','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('qualified-prefix','campaign-entry','classification-qualification'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();root=args.root.resolve(strict=True)
    if args.output.exists()or not args.output.resolve().is_relative_to(root):raise ValueError('Fresh rooted CPU phase directory required')
    sys.path.insert(0,str(HERE.parent/'pressure-throughput-feedback-v1'))
    from compare_training import measured
    study,_=measured(args.study.resolve(strict=True),root)
    runtime=read(args.runtime);control=read(args.control_comparison)
    witness=read(args.project/'source-witness.json')
    if digest(args.runtime)!=witness['runtime_manifest_sha256'] or digest(args.runtime)!=study['training_receipt']['closure']['runtime_manifest_sha256']:
        raise ValueError('Exact captured CPU/training runtime required')
    if not(study['status']=='completed'and study['optimizer_updates']==5000 and all(study['artifact_parity'].values())
        and study['restore_receipt']['optimizer_updates']==0 and not study['restore_receipt']['execution']['cuda_initialized']):
        raise ValueError('Completed CUDA5000/fresh exact CPU restore required')
    if (control['status']!='passed'or not control['campaign_package_weights_bound']
        or control['sources'].get(str(args.regression_template.resolve()))!=digest(args.regression_template)
        or control['sources'].get(str(args.control_campaign.resolve()))!=digest(args.control_campaign)):
        raise ValueError('Exact retained-control campaign/config comparison required')
    regression=read(args.regression_template);timing=read(args.timing_template)
    if not(regression['scene_indices']==list(range(74))and regression['qualify_full_episode']
        and regression['worker_memory_bytes']==2*1024**3 and regression['reserve_bytes']==8*1024**3):
        raise ValueError('Unchanged full74 memory-bounded development protocol required')
    if digest(Path(regression['runtime_manifest']))!=digest(args.runtime) or digest(Path(timing['runtime_manifest']))!=digest(args.runtime):
        raise ValueError('Regression/timing runtime differs from captured runtime')
    prefix=qualified_prefix(args,study)
    become_subreaper()
    args.output.mkdir(parents=True);started=time.perf_counter();phases=[]
    env={**os.environ,'PYTHONPATH':runtime['site_packages'],'CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    manifest=prefix['manifest']if prefix else args.output/'package/manifest.json';source=HERE.parent/'pressure-throughput-feedback-v1'
    campaign_entry=args.campaign_entry if prefix else source/'benchmark.py'
    state=dict(status='running',phase='package',training_run_id=study['training_run_id'],restore_run_id=study['restore_run_id'],
        weights_sha256=study['weights_sha256'],optimizer_updates=0,holdouts_consumed=0,candidate_promoted=False)
    if prefix:state.update(completed_prefix=str(prefix['prefix']),classification_qualification_sha256=prefix['qualification_sha256'])
    owners=[Path(__file__).resolve(),HERE/'process_ownership.py',source/'benchmark.py',source/'episode.py',source/'failure.py']
    if prefix:owners.extend([HERE/'prefix.py',HERE/'campaign.py',HERE/'rejection.py',HERE/'qualify_rejection.py'])
    measurement_sources={str(path):digest(path)for path in owners}
    write(args.output/'measurement-sources.json',dict(sources=measurement_sources,
        scope='Canonical numerical owners plus explicit current-schema exception-classification overlay; episode active_sources alone does not describe injected classification.'))
    state['measurement_sources_sha256']=digest(args.output/'measurement-sources.json')
    write(args.output/'status.json',state)
    def run(label,command,cpu=False,allow_failure=False):
        begin=time.perf_counter();state.update(phase=label);write(args.output/'status.json',state);child=None
        custody=None
        try:
            with(args.output/(label+'.log')).open('x')as log:
                child=subprocess.Popen(command,cwd=root,env=env if cpu else os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                write(args.output/'active-child.json',dict(phase=label,pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=command))
                code=child.wait()
        except BaseException:
            if child is not None:
                custody=cancel_and_drain(child)
                write(args.output/(label+'-cancel-custody.json'),custody)
            raise
        finally:
            if child is not None and child.poll()is None:
                custody=cancel_and_drain(child)
                write(args.output/(label+'-cancel-custody.json'),custody)
        custody=drain_adopted()
        if custody['signals_sent']:
            write(args.output/(label+'-unexpected-descendants.json'),custody)
            raise RuntimeError('Phase owner left live descendants; drained before stopping')
        if any(digest(Path(path))!=sha for path,sha in measurement_sources.items()):raise ValueError('CPU measurement owner changed during phase')
        row=dict(measurement_sources_sha256=state['measurement_sources_sha256'],phase=label,command=command,child_custody=custody,elapsed_s=time.perf_counter()-begin,returncode=code,
            sources={value:digest(Path(value))for value in command if value.endswith('.py')and Path(value).is_file()})
        phases.append(row);write(args.output/'phase-clocks.json',phases)
        print(json.dumps({k:row[k]for k in('phase','elapsed_s','returncode')}),flush=True)
        if code and not allow_failure:raise RuntimeError(label+' failed; retained log, no automatic retry')
        return code
    try:
        if prefix:
            fixture_passed=True
            regression['manifest']=str(manifest);timing['manifest']=str(manifest)
            write(args.output/'regression-config.json',regression);write(args.output/'timing-config.json',timing)
        else:
            run('package',[str(args.runtime_owner.parent.parent/'app-env/bin/python3.12'),'-B',str(source/'package.py'),
                '--project',str(args.project),'--study',str(args.study),'--runtime',str(args.runtime),'--runtime-owner',str(args.runtime_owner),
                '--iteration',str(HERE.parent/'water-stability-learning-v1/learned_iteration.py'),
                '--output',str(args.output/'package'),'--archive',str(args.output/'package-originals')])
            regression['manifest']=str(manifest);timing['manifest']=str(manifest)
            write(args.output/'regression-config.json',regression);write(args.output/'timing-config.json',timing)
            command=['taskset','-c','23',runtime['python'],'-B',str(Path(__file__).resolve()),'fixture',
                '--root',str(root),'--bindings',regression['bindings'],'--binary',regression['binary'],'--manifest',str(manifest),
                '--output',str(args.output/'first-frame'),'--scene','0','--fixture','--native-timeout-seconds',str(regression.get('native_timeout_seconds',300))]
            code=run('first-frame',command,cpu=True,allow_failure=True)
            result=read(args.output/'first-frame/episode.json')
            fixture_passed=code==0 and result['status']=='passed'
            if not fixture_passed and not(result.get('failure_class')=='scientific_rejection'and result.get('native_children_drained')is True):
                raise RuntimeError('First-frame infrastructure/custody failure; inspect retained result')
            run('transfer',['taskset','-c','23',runtime['python'],'-B',str(HERE.parent/'pressure-current-feedback-weight-v1/transfer.py'),
                '--project',str(args.project),'--artifacts',str(args.study/'restore'),'--output',str(args.output/'transfer.json')],cpu=True)
            run('common-rhs',[runtime['python'],'-B',str(HERE.parent/'pressure-current-policy-v1/common_rhs.py'),
                '--root',str(root),'--config',str(HERE.parent/'pressure-current-policy-v1/configs/preparation.json'),
                '--preparation',str(root/'runs/pressure-current-policy-v1/preparation-02'),'--project',str(args.project),
                '--artifacts',str(args.study/'restore'),'--output',str(args.output/'common-rhs')],cpu=True)
        if fixture_passed:
            run('regression',['python3','-B',str(campaign_entry),'--config',str(args.output/'regression-config.json'),
                '--workers','16','--output',str(args.output/'regression')])
            campaign=read(args.output/'regression/receipt.json')
            if not(campaign['all_worker_children_reaped']and len(campaign['records'])==74):raise ValueError('Full scene campaign incomplete/unreaped')
            proof=dict(status='passed',source_campaign_sha256=digest(args.control_campaign),weights_sha256=control['weights_sha256'],
                comparison_sha256=digest(args.control_comparison),scope='Retained qualified control campaign reused as comparator; no new control physics execution.')
            write(args.output/'control-equivalence.json',proof)
            run('regression-summary',['python3','-B',str(source/'summarize.py'),
                '--baseline',str(LAB/'docs/plans/water-early-pressure-v1/reports/regression-summary.json'),
                '--campaign','retained-control='+str(args.control_campaign),'--campaign','full-coarse='+str(args.output/'regression/receipt.json'),
                '--equivalence','retained-control='+str(args.output/'control-equivalence.json'),'--output',str(args.output/'regression-summary.json')])
            by_scene={r['scene']:r for r in campaign['records']}
            timed_qualified=all(by_scene[index].get('status')!='failed'for index in timing['scene_indices'])
            state.update(paired_timing_admitted=timed_qualified,
                paired_timing_scope='Fixed timed scenes must finish their full development windows; no substitute easier scenes.')
            for rep in (1,2):
                cmd=['taskset','-c','22','python3','-B',str(campaign_entry),'--config',str(args.output/'timing-config.json'),
                     '--workers','1','--output',str(args.output/f'paired-timing-{rep}')]
                if rep==2:cmd+=['--baseline',str(args.output/'paired-timing-1/receipt.json')]
                if timed_qualified:run('paired-timing-'+str(rep),cmd)
                run('raw-timing-'+str(rep),['taskset','-c','22',runtime['python'],'-B',str(HERE.parent/'pressure-spatial-coarse-v1/forward_timing.py'),
                    '--project',str(args.project),'--artifacts',str(args.study/'restore'),'--output',str(args.output/f'raw-timing-{rep}.json')],cpu=True)
        state.update(status='completed',phase='done',first_frame_passed=fixture_passed,
            full_campaign_admitted=fixture_passed,elapsed_s=time.perf_counter()-started,
            qualification='first_frame_rejected'if not fixture_passed else'recorded_full_development_result')
        write(args.output/'status.json',state)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_s=time.perf_counter()-started)
        write(args.output/'status.json',state);raise
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    if len(sys.argv)>1 and sys.argv[1]=='fixture':fixture()
    else:main()
