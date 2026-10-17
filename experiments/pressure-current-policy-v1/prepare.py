"""Bounded train-only current-policy residual prefixes and independent labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import sys
import time
from threading import Event, Thread

LAB=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(LAB/'experiments/water-residual-learning-v1'),str(LAB/'experiments/water-stability-learning-v1')]
import audit
from learned_iteration import predict_proposal


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def initialize(root,config,output,old_manifest,old_payload):
    global CONFIG,ROOT,OUTPUT,ORACLE,OLD,OLD_PAYLOAD
    CONFIG=config;ROOT=Path(root);OUTPUT=Path(output);OLD=old_manifest;OLD_PAYLOAD=Path(old_payload)
    audit.initialize(ROOT/config['project'],ROOT/config['artifacts'],config['cpus'])
    spec=importlib.util.spec_from_file_location('prefix_oracle',LAB/'experiments/global-pressure-v1/verify_capture.py')
    ORACLE=importlib.util.module_from_spec(spec);spec.loader.exec_module(ORACLE)
    vm=int(next(x.split()[1]for x in Path('/proc/self/status').read_text().splitlines()if x.startswith('VmSize:')))*1024
    resource.setrlimit(resource.RLIMIT_AS,(vm+config['worker_growth_bytes'],vm+config['worker_growth_bytes']))


def available():
    return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024


def monitor_memory(done,memory,reserve,notify,observe=available):
    try:
        while not done.wait(.1):
            memory['minimum_available_bytes']=min(memory['minimum_available_bytes'],observe())
            if memory['minimum_available_bytes']<reserve:
                if not done.is_set():notify()
                return
    except Exception as error:
        memory['observation_error']=f'{type(error).__name__}: {error}'
        if not done.is_set():notify()


def destination(root,path):
    result=path.resolve()
    if result.exists()or not result.is_relative_to(root):raise ValueError('fresh rooted output required')
    return result


def norm(value):return float(audit.TORCH.linalg.vector_norm(value))
def mean_free(value):return value-value.mean()


def label(residual,sample):
    import numpy as np
    compatible=mean_free(residual);target=audit.spectral_correction(compatible,sample)
    action=ORACLE.full_action(target.flatten().tolist(),list(target.shape[2:][::-1]),
        [float(sample['cell_size_m'][0])]*3,float(sample['rho'][0]))
    action=audit.TORCH.from_numpy(np.asarray(action,dtype='<f8')).reshape_as(target)
    defect=norm(action-compatible)/max(norm(compatible),1e-300)
    if not defect<1e-9:raise ValueError('independent current-prefix pressure label failed')
    return compatible,target,defect


def direction_key(identity):
    if identity['kind']=='real':return ('real',identity['trajectory_id'],identity['frame'])
    return ('manufactured',tuple(identity['grid']['counts_xyz']),identity['grid']['cells_per_metre'],identity['pattern'])


def error_bands(prediction,target):
    shape=target.shape[2:];basis=audit.BASES[shape][0]
    signal=audit.cosine_transform(mean_free(target)[0,0],basis)
    error=audit.cosine_transform(mean_free(prediction-target)[0,0],basis)
    total=norm(signal);torch=audit.TORCH
    z,y,x=torch.meshgrid(*[torch.arange(n)for n in shape],indexing='ij')
    low=(z<=3)&(y<=3)&(x<=3);dc=(z==0)&(y==0)&(x==0)
    amplitude=signal.abs()/max(total,1e-300);rows=[]
    for band,mask in [('low',low&~dc),('high',~low)]:
        for lower,upper in [(0.,1e-6),(1e-6,1e-4),(1e-4,.01),(.01,2.)]:
            selected=mask&(amplitude>=lower)&(amplitude<upper)
            rows.append({'band':band,'coefficient_over_full_l2_interval':[lower,upper],
                'modes':int(selected.sum()),'signal_energy_pa2':float(signal[selected].square().sum()),
                'error_energy_pa2':float(error[selected].square().sum())})
    return rows


def probe(sample,amplitude):
    if available()<CONFIG['minimum_available_bytes']:raise MemoryError('available preparation memory reserve reached')
    torch=audit.TORCH;rhs,target,defect=label(sample['native_input_residual'],sample)
    state={**sample,'base':torch.zeros_like(target),'native_input_residual':rhs,'rhs':rhs,'reference':target}
    first=predict_proposal(audit.MODEL,audit.NETWORK,state,audit.NORM,True)
    second=predict_proposal(audit.MODEL,audit.NETWORK,{**state,'base':first,
        'native_input_residual':rhs-audit.MODEL.full_operator(first,state['cell_size_m'],state['rho'])},audit.NORM,True)
    rms=float(target.square().mean().sqrt());gain=amplitude/rms
    scaled={**state,'native_input_residual':rhs*gain}
    scaled_first=predict_proposal(audit.MODEL,audit.NETWORK,scaled,audit.NORM,True)
    scaled_second=predict_proposal(audit.MODEL,audit.NETWORK,{**scaled,'base':scaled_first,
        'native_input_residual':rhs*gain-audit.MODEL.full_operator(scaled_first,state['cell_size_m'],state['rho'])},audit.NORM,True)/gain
    error=norm(mean_free(second-target))/max(norm(mean_free(target)),1e-300)
    result={'target_rms_pa':rms,'independent_operator_relative_defect':defect,
        'one_call_mean_free_pressure_relative_l2':norm(mean_free(first-target))/norm(mean_free(target)),
        'two_call_mean_free_pressure_relative_l2':error,
        'two_call_relative_residual':norm(rhs-audit.MODEL.full_operator(second,state['cell_size_m'],state['rho']))/norm(rhs),
        'rescale_gain_to_train_rms':gain,'rescaled_two_call_mean_free_pressure_relative_l2':norm(mean_free(scaled_second-target))/norm(mean_free(target)),
        'native_vs_rescaled_prediction_relative_l2':norm(mean_free(second-scaled_second))/norm(mean_free(target)),
        'frequency_amplitude':error_bands(second,target)}
    return result,rhs,target


def prepare(task):
    import numpy as np
    torch=audit.TORCH;index,identity,arrays,amplitude=task;started=time.perf_counter()
    sample={k:torch.from_numpy(v.copy())for k,v in arrays.items()};initial=sample['native_input_residual'];base=sample['base'];pressure=base.clone()
    folder=OUTPUT/f'case-{index:03d}';folder.mkdir();rows=[];fields={};calls=0
    with torch.no_grad():
        early,early_rhs,early_target=probe(sample,amplitude);calls+=4
        fields.update(early_rhs=early_rhs.numpy().copy(),early_reference=early_target.numpy().copy())
        for prefix in range(1,max(CONFIG['prefixes'])+1):
            residual=initial if prefix==1 else initial-audit.MODEL.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
            pressure=predict_proposal(audit.MODEL,audit.NETWORK,{**sample,'base':pressure,'native_input_residual':residual},audit.NORM,True);calls+=1
            if available()<CONFIG['minimum_available_bytes']:raise MemoryError('available prefix memory reserve reached')
            if prefix not in CONFIG['prefixes']:continue
            residual=initial-audit.MODEL.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
            ratio=norm(residual)/max(norm(initial),1e-300)
            row={'prefix':prefix,'residual_over_initial_l2':ratio}
            if ratio<=CONFIG['minimum_late_ratio']:
                rows.append({**row,'excluded':'paired late/initial residual numerical guard'});continue
            measured,rhs,target=probe({**sample,'native_input_residual':residual},amplitude);calls+=4
            identity_error=norm(mean_free(early_target-(pressure-base)-target))/max(norm(mean_free(early_target)),1e-300)
            if not identity_error<1e-9:raise ValueError('remaining correction identity failed')
            row.update(measured,remaining_identity_relative_initial_pressure=identity_error)
            if prefix==CONFIG['bank_prefix']:
                fields.update(late_rhs=rhs.numpy().copy(),late_reference=target.numpy().copy())
                matches=[(i,case)for i,case in enumerate(OLD['cases'])if direction_key(case['identity'])==direction_key(identity)]
                if len(matches)!=1:raise ValueError('one unique prior-bank training direction required')
                old_index,_=matches[0]
                with np.load(OLD_PAYLOAD,allow_pickle=False)as values:old=torch.from_numpy(values[f'case_{old_index:03d}_reference'].copy())
                current=mean_free(target);old=mean_free(old);cosine=float((current*old).sum())/(norm(current)*norm(old))
                row['old_scaled_bank_direction_cosine']=cosine
                row['unit_direction_difference_from_old']=norm(current/norm(current)-old/norm(old))
            rows.append(row)
    paired='late_rhs'in fields
    if paired:
        with (folder/'paired.npz').open('xb')as stream:np.savez(stream,**fields)
    record={'identity':identity,'counts_xyz':list(base.shape[2:][::-1]),
        **{k:float(sample[k][0])for k in ('rho','dt','cell_size_m')},'early':early,'prefixes':rows,'paired_bank_eligible':paired,
        'pair_path':str((folder/'paired.npz').relative_to(OUTPUT))if paired else None,
        'pair_sha256':digest(folder/'paired.npz')if paired else None,'network_calls':calls,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_initialized':torch.cuda.is_initialized(),
        'cpu_threads':torch.get_num_threads(),'elapsed_s':time.perf_counter()-started,
        'worker_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    assert not record['cuda_initialized'];write(folder/'receipt.json',record);audit.BASES.clear();return record


def bound_inputs(root,config):
    project=root/config['project']
    witness=json.loads((project/'source-witness.json').read_text())
    document=json.loads((project/'project.json').read_text())
    refs={row['id']:row for row in document['data']};inputs={};identities={}
    for slot,expected in witness['inputs'].items():
        path=(root/refs[slot]['path']).resolve(strict=True)
        if not path.is_relative_to(root):raise ValueError('captured dataset outside selected artifact root')
        raw=path.read_bytes()
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256'] or refs[slot]['sha256']!=expected['sha256']:
            raise ValueError('dataset differs from actual checkpoint captured input: '+slot)
        inputs[slot]=raw;identities[slot]={'path':str(path),**expected}
    return inputs,identities


def source_identities(root,config):
    import learned_iteration
    sources=[Path(__file__),Path(audit.__file__),Path(learned_iteration.__file__),
        LAB/'experiments/global-pressure-v1/verify_capture.py',root/config['project']/'analytical.py']
    return {str(p.resolve()):digest(p)for p in sources}


def tasks(root,config,fixture):
    import numpy as np
    import analytical
    inputs,_=bound_inputs(root,config)
    _,parents=audit.DATA.load((inputs['dataset'],inputs['dataset_extra']),json.loads(inputs['splits']))
    real=[(p,i,audit.DATA.sample(p,i))for p in parents if p['entry']['split']=='train'for i in range(len(p['frames']))]
    assert len(real)==32 and len({p['entry']['trajectory_id']for p,_,_ in real})==16
    amplitude=analytical.training_correction_rms(s for _,_,s in real)
    def task(index,identity,sample):
        return index,identity,{k:sample[k].numpy().copy()for k in ('base','native_input_residual','rho','dt','cell_size_m')},amplitude
    index=0
    for parent,slot,sample in real[:1]if fixture else real:
        if available()<config['minimum_available_bytes']+config['worker_reservation_bytes']:raise MemoryError('memory reserve before dispatch')
        yield task(index,{'kind':'real','group':'real','trajectory_id':parent['entry']['trajectory_id'],'frame':parent['frames'][slot]['index']},sample);index+=1
    source=json.loads((root/config['project']/'config.json').read_text())
    for bank in ('analytical_feedback','supplemental_feedback'):
        specification=source[bank]
        for grid,detail in enumerate(specification['grids']):
            template={**real[0][2],'base':audit.TORCH.zeros((1,1,*detail['counts_xyz'][::-1]),dtype=audit.TORCH.float64),
                'cell_size_m':audit.TORCH.tensor([1/detail['cells_per_metre']],dtype=audit.TORCH.float64)}
            values,_,_=analytical.single_grid_bank(template,{**specification,'patterns':config['patterns_per_grid']},1,amplitude=amplitude)
            for pattern in range(config['patterns_per_grid']):
                identity={'kind':'manufactured','group':f'{bank}-{grid:02d}','grid':detail,'pattern':pattern,'source_amplitude_slot':config['source_amplitude_slot']}
                if available()<config['minimum_available_bytes']+config['worker_reservation_bytes']:raise MemoryError('memory reserve before dispatch')
                yield task(index,identity,values[pattern*len(analytical.AMPLITUDES)+config['source_amplitude_slot']]);index+=1
                if fixture:return
            del values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fixture',action='store_true');a=p.parse_args();root=a.root.resolve(strict=True);config=json.loads(a.config.read_text());output=destination(root,a.output)
    if config['prefixes']!=[2,4,8,12]or config['bank_prefix']!=8 or not 1<=config['workers']<=16:raise ValueError('declared prefix and bounded worker contract required')
    config['cpus']=[cpu for cpu in config['cpus']if cpu in os.sched_getaffinity(0)]
    workers=min(config['workers'],len(config['cpus']),2 if a.fixture else 16)
    available_start=available()
    if not workers or available_start<workers*config['worker_reservation_bytes']+config['minimum_available_bytes']:raise ValueError('insufficient available CPU/memory reservation')
    inputs,input_identities=bound_inputs(root,config);sources=source_identities(root,config)
    output.mkdir(parents=True);started=time.perf_counter();audit.initialize(root/config['project'],root/config['artifacts'],config['cpus']);os.sched_setaffinity(0,config['cpus'])
    document=json.loads((root/config['project']/'project.json').read_text());refs={r['id']:r for r in document['data']};old_path=root/refs['direction_manifest']['path'];old=json.loads(old_path.read_text());payload=root/refs['direction_dataset']['path']
    assert digest(old_path)==refs['direction_manifest']['sha256']and digest(payload)==refs['direction_dataset']['sha256']
    if available()<workers*config['worker_reservation_bytes']+config['minimum_available_bytes']:raise MemoryError('insufficient memory after model initialization')
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(root),config,str(output),old,str(payload)))
    memory={'minimum_available_bytes':available()}
    done=Event()
    def interrupted(*_):
        if memory.get('observation_error'):raise RuntimeError('memory observation unavailable: '+memory['observation_error'])
        if memory['minimum_available_bytes']<config['minimum_available_bytes']:raise MemoryError('monitored preparation memory reserve reached')
        raise InterruptedError('prefix owner interrupted; drain bounded active workers')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    thread=Thread(target=monitor_memory,args=(done,memory,config['minimum_available_bytes'],lambda:os.kill(os.getpid(),signal.SIGTERM)),name='prefix-memory-monitor');thread.start()
    try:
        records=list(pool.map(prepare,tasks(root,config,a.fixture),buffersize=workers))
        pool.shutdown(wait=True)
    except BaseException as error:
        done.set();signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        pool.shutdown(wait=True,cancel_futures=True);write(output/'failure.json',{'status':'failed','error':str(error),'workers_drained':True,'elapsed_s':time.perf_counter()-started});raise
    finally:done.set();thread.join(timeout=5)
    if thread.is_alive()or memory.get('observation_error'):raise RuntimeError('memory observation did not complete cleanly')
    receipt={'format':'pressure_current_policy_preparation_v1','status':'passed','configuration':config,'fixture':a.fixture,'cases':records,
        'weights_sha256':digest(root/config['artifacts']/'weights.npz'),'normalization_sha256':digest(root/config['artifacts']/'normalization.json'),
        'source_witness_sha256':digest(root/config['project']/'source-witness.json'),'packet_splits_sha256':hashlib.sha256(inputs['splits']).hexdigest(),'captured_inputs':input_identities,'execution_source_sha256s':sources,
        'prior_direction_manifest_sha256':digest(old_path),'source_sha256':digest(__file__),'workers':workers,'all_workers_reaped':True,
        'train_parent_count':16,'real_train_states':32,'network_calls':sum(r['network_calls']for r in records),'elapsed_s':time.perf_counter()-started,
        'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'available_at_start_bytes':available_start,'memory':memory,
        'optimizer_updates':0,'simulations_advanced':0,'scope':'Training parents and admitted analytic patterns only; physical cubic operator. Common-RHS rescaling test distinguishes amplitude from direction. Native amplitudes retained. No holdout or inference change.'}
    if sources!=source_identities(root,config):raise ValueError('executed diagnostic source changed during preparation')
    write(output/'receipt.json',receipt);print(json.dumps({k:receipt[k]for k in ('status','workers','network_calls','elapsed_s')}))

if __name__=='__main__':main()
