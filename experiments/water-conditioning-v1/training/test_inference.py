"""Optimizer-free validation or reserved inference with typed managed checkpoint custody."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from policy import HERE, MIB, canonical, decode, read, require, sha, write
import data
import model
import train

TRAINED = {'trained_weights':('weights.npz','checkpoint',2*MIB),
    'trained_normalization':('normalization.json','preprocessing',MIB),
    'trained_receipt':('receipt.json','training_result',MIB),
    'trained_history':('history.json','training_diagnostics',MIB)}
OUTPUTS = {'predictions.npz':('predictions',2*MIB),'receipt.json':('evaluation_result',MIB),'summary.json':('result_summary',65536)}


def definition(config, sources, snapshot, prefix, population):
    require(population in ('validation','reserved_test'), 'population', 'explicit CPU inference population')
    inputs = {population+'_'+slot:{'max_bytes':4*MIB if slot=='splits' else 32*MIB} for slot in data.DATA}
    inputs.update({slot:{'max_bytes':cap} for slot,(_name,_role,cap) in TRAINED.items()})
    arguments = [{'literal':'--population'},{'literal':population}]
    for slot in inputs: arguments.extend([{'literal':'--'+slot.replace('_','-')},{'input':slot}])
    for name in ('runtime-manifest','source-witness'): arguments.extend([{'literal':'--'+name},{'source':prefix+name+'.json'}])
    arguments.extend([{'literal':'--input-evidence'},{'input_evidence':True},{'literal':'--output-dir'},{'output_dir':True}])
    return {'schema_version':1,'format':'external_recipe_v1','id':config['id']+'-'+population,'name':config['id']+' · '+population+' CPU',
        'version':1,'runtime':'managed_torch_python_v1','runtime_device':'cpu','runtime_snapshot_sha256':snapshot,
        'sources':[{'path':prefix+name,'sha256':sha(raw)} for name,raw in sorted(sources.items())],
        'entrypoint':prefix+'test_inference.py','arguments':arguments,'parameters':{},'inputs':inputs,
        'outputs':[{'path':name,'role':role,'max_bytes':cap,'required':True} for name,(role,cap) in OUTPUTS.items()],
        'environment':{},'resources':config['resources']}


def run(args):
    start = time.perf_counter(); witness_raw = read(args.source_witness, 2*MIB); witness = decode(witness_raw)
    require(witness['schema']=='rheon-water-learning-source-v1', 'source', 'versioned selected-frame experiment')
    for name,digest in witness['sources'].items(): require(sha(read(HERE/name,4*MIB))==digest,'source_identity',name)
    runtime_raw = read(args.runtime_manifest,MIB); runtime = decode(runtime_raw)
    require(sha(runtime_raw)==witness['runtime_manifest_sha256'] and runtime['profile']=='managed_torch_python_v1'
        and runtime['packages']['torch']==torch.__version__ and runtime['packages']['numpy']==np.__version__
        and os.environ.get('TULDOK_TORCH_DEVICE')=='cpu','runtime','qualified CPU inference')
    model.configure('cpu')
    model.select(decode(read(HERE/'config.json')))
    prefix = args.population+'_'
    inputs = {prefix+slot:read(getattr(args,prefix+slot),4*MIB if slot=='splits' else 32*MIB) for slot in data.DATA}
    evidence = decode(read(args.input_evidence,MIB)); require(set(evidence)==set(inputs)|set(TRAINED),'inputs','complete typed evaluation inputs')
    train.project_inputs(evidence,inputs,witness)
    raw_trained = {slot:read(getattr(args,slot),cap) for slot,(_name,_role,cap) in TRAINED.items()}
    origin = train.custody(evidence,raw_trained,TRAINED,HERE/'custody-validator.py',witness['custody_validator_sha256'])
    trained = decode(raw_trained['trained_receipt']); normalization = decode(raw_trained['trained_normalization']); history = decode(raw_trained['trained_history'])
    require(trained['schema']=='rheon-water-learning-training-v1' and trained['mode']=='train'
        and trained['closure']['source_witness_sha256']==sha(witness_raw) and trained['closure']['inputs']==witness['inputs']
        and trained['closure']['normalization_sha256']==sha(raw_trained['trained_normalization'])==witness['normalization_sha256'], 'candidate', 'same predeclared frozen final candidate')
    for slot,(name,_role,_cap) in TRAINED.items():
        if name!='receipt.json': require(trained['outputs'][name]['sha256']==sha(raw_trained[slot]),'artifact_hash',name)
    config = decode(read(HERE/'config.json')); final=f'checkpoint-{config["training"]["updates"]:03d}.npz'
    require(trained['optimizer_updates']==config['training']['updates'] and history['checkpoints'][final]['sha256']==sha(raw_trained['trained_weights']), 'final_candidate', 'declared terminal state only')
    network = model.PressureCorrectionNet(); data.helpers().load_weights(network,args.trained_weights,history['checkpoints'][final]['inventory'])
    reservation,parents = data.load((inputs[prefix+'dataset'],inputs[prefix+'dataset_extra']),decode(inputs[prefix+'splits']))
    require(sha(canonical(reservation))==witness['evaluation_reservations'][args.population] and all(p['entry']['split']==args.population for p in parents), 'evaluation_population', 'exact independent population')
    cases = [{'parent':i,'frame':slot,'native_frame':frame['index'],'trajectory_id':parent['entry']['trajectory_id'],'split':args.population,
        'counts_xyz':parent['counts_xyz'],'cell_size_m':parent['cell_size_m']} for i,parent in enumerate(parents) for slot,frame in enumerate(parent['frames'])]
    predicted,clocks = train.predictions(network,parents,normalization,cases); inventory=data.inventory(predicted)
    require(sum(v.nbytes for v in predicted.values())<2*MIB,'output_budget','bounded complete evaluation predictions')
    output=args.output_dir.absolute();require(output==output.resolve() and output.is_dir() and not any(output.iterdir()),'output','fresh managed output')
    with (output/'predictions.npz').open('xb') as stream:np.savez(stream,**predicted)
    require(not torch.cuda.is_initialized(),'cpu_only','zero CUDA initialization')
    receipt = {'schema':'rheon-water-learning-inference-v1','status':'complete','population':args.population,'optimizer_updates':0,
        'cases':cases,'prediction_inventory':inventory,'source_witness_sha256':sha(witness_raw),'weights_sha256':sha(raw_trained['trained_weights']),
        'normalization_sha256':sha(raw_trained['trained_normalization']),'originating_training_run':origin,
        'runtime':{'device':'cpu','torch':torch.__version__,'numpy':np.__version__,'cuda_initialized':False},
        'predictions_sha256':sha(read(output/'predictions.npz',2*MIB)),'per_case_inference_s':clocks,'elapsed_s':time.perf_counter()-start}
    write(output/'receipt.json',receipt);write(output/'summary.json',{'status':'inference_complete','population':args.population,'parents':len(parents),'cases':len(cases),'optimizer_updates':0,'accuracy_claim':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--population',choices=('validation','reserved_test'),required=True)
    known,_=parser.parse_known_args()
    for name in (*(known.population+'_'+slot for slot in data.DATA),*TRAINED,'runtime_manifest','source_witness','input_evidence','output_dir'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
