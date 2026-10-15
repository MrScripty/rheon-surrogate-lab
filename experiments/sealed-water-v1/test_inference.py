"""Single final reserved-test inference, owned by a normal managed CPU Run."""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from policy import HERE,MIB,canonical,decode,read,require,sha,write
import data
import model
import train

TRAINED={'trained_weights':('weights.npz','checkpoint',2*MIB),
         'trained_normalization':('normalization.json','preprocessing',MIB),
         'trained_receipt':('receipt.json','training_result',MIB),
         'trained_history':('history.json','training_diagnostics',MIB)}
OUTPUTS={'predictions.npz':('predictions',2*MIB),'receipt.json':('evaluation_result',MIB),'summary.json':('result_summary',65536)}


def definition(config,sources,snapshot,prefix):
    inputs={'test_dataset':{'max_bytes':32*MIB},'test_splits':{'max_bytes':4*MIB},**{slot:{'max_bytes':cap} for slot,(_,_,cap) in TRAINED.items()}}
    arguments=[]
    for slot in inputs:arguments.extend([{'literal':'--'+slot.replace('_','-')},{'input':slot}])
    for name in ('runtime-manifest','source-witness'):arguments.extend([{'literal':'--'+name},{'source':prefix+name+'.json'}])
    arguments.extend([{'literal':'--input-evidence'},{'input_evidence':True},{'literal':'--output-dir'},{'output_dir':True}])
    return {'schema_version':1,'format':'external_recipe_v1','id':'sealed-water-reserved-test','name':'Frozen water candidate · reserved CPU test','version':1,
        'runtime':'managed_torch_python_v1','runtime_device':'cpu','runtime_snapshot_sha256':snapshot,
        'sources':[{'path':prefix+n,'sha256':sha(v)} for n,v in sorted(sources.items())],'entrypoint':prefix+'test_inference.py',
        'arguments':arguments,'parameters':{},'inputs':inputs,'outputs':[{'path':n,'role':r,'max_bytes':c,'required':True} for n,(r,c) in OUTPUTS.items()],
        'environment':{},'resources':config['resources']}


def run(args):
    start=time.perf_counter();witness_raw=read(args.source_witness,2*MIB);witness=decode(witness_raw)
    require(witness['schema']=='rheon-sealed-water-source-v1','source','versioned water source')
    for name,digest in witness['sources'].items():require(sha(read(HERE/name,4*MIB))==digest,'source_identity',name)
    runtime_raw=read(args.runtime_manifest,MIB);runtime=decode(runtime_raw)
    require(sha(runtime_raw)==witness['runtime_manifest_sha256'] and runtime['profile']=='managed_torch_python_v1'
        and runtime['packages']['torch']==torch.__version__ and runtime['packages']['numpy']==np.__version__
        and runtime['python_version']==f'{sys.version_info.major}.{sys.version_info.minor}'
        and os.environ.get('TULDOK_TORCH_DEVICE')=='cpu','runtime','approved CPU inference')
    model.configure('cpu');inputs={'test_dataset':read(args.test_dataset,32*MIB),'test_splits':read(args.test_splits,4*MIB)}
    evidence=decode(read(args.input_evidence,MIB));require(set(evidence)==set(inputs)|set(TRAINED),'inputs','complete typed CPU test inputs')
    train.project_inputs(evidence,inputs,witness)
    raw_trained={slot:read(getattr(args,slot),cap) for slot,(_,_,cap) in TRAINED.items()}
    origin=train.custody(evidence,raw_trained,TRAINED,HERE/'custody-validator.py',witness['custody_validator_sha256'])
    trained=decode(raw_trained['trained_receipt']);normalization=decode(raw_trained['trained_normalization']);history=decode(raw_trained['trained_history'])
    require(trained['schema']=='rheon-sealed-water-training-v1' and trained['mode']=='train'
        and trained['closure']['source_witness_sha256']==sha(witness_raw) and trained['closure']['inputs']==witness['inputs']
        and trained['closure']['normalization_sha256']==sha(raw_trained['trained_normalization'])==witness['normalization_sha256'],
        'candidate','same predeclared frozen final training candidate')
    for slot,(name,_role,_cap) in TRAINED.items():
        if name!='receipt.json':require(trained['outputs'][name]['sha256']==sha(raw_trained[slot]),'artifact_hash',name)
    config=decode(read(HERE/'config.json'));final=f'checkpoint-{config["training"]["updates"]:03d}.npz'
    require(trained['optimizer_updates']==config['training']['updates'] and history['checkpoints'][final]['sha256']==sha(raw_trained['trained_weights']),
        'final_candidate','final checkpoint chosen before reserved inference')
    network=model.PressureCorrectionNet();data.helpers().load_weights(network,args.trained_weights,history['checkpoints'][final]['inventory'])
    reservation,parents=data.load(inputs['test_dataset'],decode(inputs['test_splits']))
    require(sha(canonical(reservation))==witness['packet_experiment_sha256'] and all(p['entry']['split']=='reserved_test' for p in parents),
        'reserved_test','whole independent reserved simulations only')
    cases=[{'parent':i,'frame':slot,'native_frame':frame['index'],'trajectory_id':p['entry']['trajectory_id'],'split':'reserved_test',
        'counts_xyz':p['counts_xyz'],'cell_size_m':p['cell_size_m']} for i,p in enumerate(parents) for slot,frame in enumerate(p['frames'])]
    predicted,clocks=train.predictions(network,parents,normalization,cases);inventory=data.inventory(predicted)
    require(sum(v.nbytes for v in predicted.values())<2*MIB,'output_budget','bounded reserved predictions')
    output=args.output_dir.absolute();require(output==output.resolve() and output.is_dir() and not any(output.iterdir()),'output','fresh managed output')
    with (output/'predictions.npz').open('xb') as stream:np.savez(stream,**predicted)
    receipt={'schema':'rheon-sealed-water-reserved-inference-v1','status':'complete','optimizer_updates':0,'cases':cases,'prediction_inventory':inventory,
        'source_witness_sha256':sha(witness_raw),'weights_sha256':sha(raw_trained['trained_weights']),'normalization_sha256':sha(raw_trained['trained_normalization']),
        'originating_training_run':origin,'runtime':{'device':'cpu','torch':torch.__version__,'numpy':np.__version__,'cuda_initialized':torch.cuda.is_initialized()},
        'predictions_sha256':sha(read(output/'predictions.npz',2*MIB)),'per_case_inference_s':clocks,'elapsed_s':time.perf_counter()-start}
    require(not torch.cuda.is_initialized(),'cpu_only','zero CUDA initialization')
    write(output/'receipt.json',receipt);write(output/'summary.json',{'status':'reserved_inference_complete','parents':len(parents),'cases':len(cases),'optimizer_updates':0,'accuracy_claim':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('test_dataset','test_splits',*TRAINED,'runtime_manifest','source_witness','input_evidence','output_dir'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
