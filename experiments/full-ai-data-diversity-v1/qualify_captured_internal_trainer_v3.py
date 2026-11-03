"""Actual captured compact/cache trainer initial bytes, loss and gradients; zero updates."""
import argparse,hashlib,importlib,json,math,os
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh CUDA-only captured trainer qualification required')
    from managed_input_stream_v1 import streamed_identity
    reg=read(args.protocol)
    for name,h in reg['source_sha256'].items():
        if streamed_identity(ROOT/name)['sha256']!=h:raise ValueError('Captured trainer qualification source changed')
    project=ROOT/reg['project'];assembly=read(project/'assembly.json');config=assembly['config']
    for row in assembly['sources']:
        if streamed_identity(ROOT/row['path'])['sha256']!=row['sha256']:raise ValueError('Captured source changed')
    for row in assembly['static_inputs'].values():
        if streamed_identity(ROOT/row['path'])!={k:row[k] for k in ('bytes','sha256')}:raise ValueError('Captured typed input changed')
    proof=read(ROOT/reg['regeneration_proof']);fingerprints={(r['identifier'],r['start'],r['group_index']):r['array_sha256'] for r in proof['rows']}
    sys.path.insert(0,str(project));trainer=importlib.import_module('train_internal_residual_v3')
    import numpy as np
    import torch
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    args.output=args.output.resolve();args.output.mkdir(parents=True);(args.output/'unused-output').mkdir()
    static={n:ROOT/r['path'] for n,r in assembly['static_inputs'].items()}
    from types import SimpleNamespace
    actual=SimpleNamespace(config=project/'config.json',manifest=static['manifest'],normalization=static['normalization'],optimization_normalization=static['optimization_normalization'],runtime_manifest=static['runtime_manifest'],teacher_code=project/'teacher-code.zip',shards=[static['part_00']],output_dir=args.output/'unused-output')
    manifest=trainer.read(actual.manifest);model=trainer.InternalGroupResidual(normalization=trainer.read(actual.normalization));trainer.load_warm(model,manifest,config['qualified_initial_inventory']);before=trainer.inventory(model)
    source=cache=None;started=time.monotonic();result=dict(status='running',optimizer_updates=0,new_parameter_fits=0,cpu_model_forward=False,expanded_reserved52_references_opened=0,full_generator_target_pass=False)
    def bounded():
        if time.monotonic()-started>reg['maximum_wall_seconds'] or trainer.process_memory()['peak_since_exec_bytes']>config['maximum_peak_rss_bytes'] or torch.cuda.max_memory_allocated()>config['maximum_gpu_allocated_bytes'] or torch.cuda.max_memory_reserved()>config['maximum_gpu_reserved_bytes']:raise ValueError('Actual trainer qualification resource budget exceeded')
    try:
        source,cache=trainer.prepare_cache(actual,config,manifest)
        checked=0
        for i in range(783):
            row,sample,response=cache.sample(i);expected=fingerprints[row['identifier'],row['start'],row['group_index']]
            for name,value in sample.items():
                a=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value,dtype='<f8')
                if hashlib.sha256(a.tobytes()).hexdigest()!=expected[name]:raise ValueError('Actual cached training array differs: '+name)
                checked+=1
            bounded()
        observed=trainer.full_objective(model,cache,trainer.read(actual.optimization_normalization),True,bounded)
        if not math.isclose(observed['objective'],config['qualified_initial_objective'],rel_tol=1e-10,abs_tol=1e-30):raise ValueError('Captured trainer warm-start objective differs')
        gradients={}
        for n,value in model.named_parameters():
            actual_norm=float(value.grad.double().norm());expected=config['qualified_initial_gradient_norms'][n]
            if not math.isclose(actual_norm,expected,rel_tol=1e-6,abs_tol=1e-16):raise ValueError('Captured trainer gradient differs: '+n)
            gradients[n]=actual_norm
        if trainer.inventory(model)!=before:raise ValueError('Zero-update captured trainer changed model weights')
        result.update(status='passed_actual_compact_cached_trainer_initial_objective_and_gradients',all148training_scenes_in_objective=True,
            cached_training_tensor_bytes=cache.cached_tensor_bytes,all_required_sample_arrays_byte_exact=True,checked_arrays=checked,
            observed=observed,gradient_norms=gradients,parameter_inventory=before,zero_update_parameters_unchanged=True,
            memory_counters=trainer.process_memory(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-started,source_sha256=reg['source_sha256'])
    except BaseException as error:result.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-started);raise
    finally:
        if source is not None:source.close()
        if cache is not None:cache.close()
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Actual captured compact/cache trainer warm-start loss, gradients and all arrays verified; no optimizer updates.',flush=True)
if __name__=='__main__':main()
