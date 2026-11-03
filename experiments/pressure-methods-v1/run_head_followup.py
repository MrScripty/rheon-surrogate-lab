"""One qualified coarse-head optimizer follow-up, then fixed scientific evaluations."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment


def run():
    base=ROOT/'runs/pressure-methods-v1'
    arm='coarse-head-rate'
    output=base/'coarse-head-rate-controller-01'
    if output.exists():
        raise ValueError('Fresh bounded follow-up required')
    qualification=ROOT/'docs/plans/pressure-methods-v1/coarse-head-rate-qualification-02.json'
    checked=read(qualification)
    if checked['status']!='passed' or checked['optimizer_updates']!=0 or not checked['coarse_head_optimizer']['disjoint_complete_groups']:
        raise ValueError('Actual objective and optimizer-group qualification required')
    output.mkdir()
    sources={name:sha(HERE/name)for name in ['manage.py','evaluate_fit.py','evaluate_extra.py','drift_episode_extended.py']}
    plan=dict(status='qualified',arm=arm,qualification_sha256=sha(qualification),updates=1600,
        changed_factor='Only added-coarse-logit learning-rate multiplier 1000; all original fine-weight rates retained',
        unchanged=['initialization','data','sampler','loss','clipping','architecture','inference gate'],
        source_sha256=sha(__file__),sources=sources,automatic_retry=False,
        native_rule='Evaluate 1/2/4/8 passes; test the lowest all-consumed128-qualified pass budget, followed by larger standard budgets only after a pressure rejection.')
    write(output/'plan.json',plan)
    state=dict(status='waiting_for_prior_sequence',phase='waiting',workers=[])
    write(output/'execution.json',state)
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    cpu=sorted(os.sched_getaffinity(0))[-7]
    process=None
    def stop(*_):
        raise KeyboardInterrupt('Coarse-head follow-up interrupted; inspect retained Run custody')
    signal.signal(signal.SIGTERM,stop)
    def worker(name,command,timeout,env=None):
        nonlocal process
        if any(sha(HERE/n)!=expected for n,expected in sources.items()):
            raise ValueError('Frozen follow-up controller source changed')
        state.update(status='running',phase=name)
        write(output/'execution.json',state)
        tick=time.perf_counter()
        with (output/(name+'.log')).open('xb')as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('Coarse-head follow-up',name,flush=True)
            process.wait(timeout=timeout)
        if process.returncode:
            raise RuntimeError('Follow-up stopped; inspect '+name)
        state['workers'].append(dict(name=name,pid=process.pid,reaped=True,returncode=0,elapsed_s=time.perf_counter()-tick))
        write(output/'execution.json',state)
    try:
        deadline=time.monotonic()+7200
        while True:
            previous=read(base/'hybrid-spacing-sequence-01/execution.json')
            if previous['status']=='completed':break
            if previous['status']=='stopped' or time.monotonic()>deadline:
                raise RuntimeError('Prior fit sequence unavailable; no follow-up admitted')
            time.sleep(2)
        study=base/(arm+'-study-01')
        project=base/(arm+'-project-01')
        command=[sys.executable,'-B',str(HERE/'manage.py')]
        for key,value in dict(config=base/(arm+'-experiment.json'),assembly=project/'assembly.json',qualification=qualification,
                              tuldok=ROOT/'runs/dependency-migration-v1/Tuldok',session_file=ROOT/'runs/tuldok-performance-v1/private/owner-session.txt',output=study).items():
            command+=['--'+key.replace('_','-'),str(value)]
        command+=['--endpoint','http://127.0.0.1:8091','--catalog-id','a7da4a8de53541a2a0e2aad08005bb71','--project-alias','current-lab']
        worker('managed-fit-and-restore',command,4200)
        evaluation=base/'evaluation/coarse-head-rate-01'
        command=[runtime['python'],'-B',str(HERE/'evaluate_fit.py'),'--project',str(project),'--study',str(study),
            '--name',arm,'--output',str(evaluation),'--repetitions','1']
        # Evaluation uses the final CPU in its inherited affinity; isolate this
        # child from the other campaign evaluation worker.
        worker('evaluation',['/usr/bin/taskset','--cpu-list',str(cpu),*command],900,environment(runtime))
        extension=output/'pass-extension.json'
        worker('pass-extension',[runtime['python'],'-B',str(HERE/'evaluate_extra.py'),'--evaluation',str(evaluation/'results.json'),
            '--population','consumed128','--passes','1','2','3','4','5','6','7','8','--cpu',str(cpu),'--output',str(extension)],900,environment(runtime))
        summary=read(extension)['summary']
        options=[r for r in summary if r['qualified']==r['cases']]
        selected=min((r['passes']for r in options),default=8)
        for passes in [selected,*[n for n in [2,4,8]if n>selected]]:
            name='native-'+str(passes)
            destination=output/name
            worker(name,[runtime['python'],'-B',str(HERE/'drift_episode_extended.py'),'--build',str(base/'drift-observation-02'),
                '--output',str(destination),'--cpu',str(cpu),'--steps','512','--seed','20261103','--kind','neural',
                '--passes',str(passes),'--fit-evaluation',str(evaluation/'results.json')],660,environment(runtime))
            result=read(destination/'results.json')
            if result['status']=='completed':break
        state.update(status='completed',phase='complete')
    except BaseException as error:
        state.update(status='stopped',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if process is not None and process.poll()is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        state['owned_worker_reaped']=process is None or process.poll()is not None
        write(output/'execution.json',state)


if __name__=='__main__':
    run()
