"""Freeze bounded selected-frame trials and existing normal managed recipes."""
import io
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch
import numpy as np
import torch
from policy import HERE, LAB, MIB, canonical, decode, execution, experiment, fresh, read, require, sha, write
import data
import model
import train


def definition(config, sources, snapshot, mode, prefix):
    inputs = {name: {'max_bytes': 4*MIB if name == 'splits' else 32*MIB} for name in train.DATA}
    arguments = [{'literal': mode}]
    for slot in inputs:
        arguments += [{'literal': '--'+slot.replace('_', '-')}, {'input': slot}]
    for name in ('config', 'runtime-manifest', 'source-witness'):
        arguments += [{'literal': '--'+name}, {'source': prefix+name+'.json'}]
    arguments += [{'literal': '--input-evidence'}, {'input_evidence': True}, {'literal': '--output-dir'}, {'output_dir': True}]
    if mode == 'restore':
        for slot, (_name, _role, cap) in train.trained_inputs(config).items():
            inputs[slot] = {'max_bytes': cap}
            arguments += [{'literal': '--'+slot.replace('_', '-')}, {'input': slot}]
    return {'schema_version': 1, 'format': 'external_recipe_v1', 'id': config['id']+'-'+mode,
        'name': config['id']+' · '+mode, 'version': 1, 'runtime': 'managed_torch_python_v1', 'runtime_device': 'cuda' if mode == 'train' else 'cpu',
        'runtime_snapshot_sha256': snapshot, 'sources': [{'path': prefix+name, 'sha256': sha(raw)} for name, raw in sorted(sources.items())],
        'entrypoint': prefix+'train.py', 'arguments': arguments, 'parameters': {}, 'inputs': inputs,
        'outputs': [{'path': name, 'role': 'evaluation_result' if mode == 'restore' and name == 'receipt.json' else role, 'max_bytes': cap, 'required': True} for name, (role, cap) in train.outputs(config).items()],
        'environment': {}, 'resources': config['resources']}


def assemble(packet, validation, reserved, runtime_manifest, tuldok, output, configuration):
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=LAB), 'source_commit', 'commit qualified experiment source before material capture')
    packet_config, parents = data.read_view(packet)
    config = experiment(decode(read(configuration, MIB)))
    model.select(config)
    if execution(config)['batch_mode']=='stacked':
        require(len({tuple(p['counts_xyz']) for p in parents if p['entry']['split']=='train'})==1,
                'batch_shape','stacked benchmark requires one grid shape; heterogeneous batching is unsupported')
    packet_receipt = decode(read(packet/'dataset-receipt.json', MIB))
    require(packet_receipt['status'] == 'verified_lossless_water_view' and packet_receipt['reader_source_sha256'] == sha(read(HERE/'data.py')), 'view_receipt', 'qualified current reader')
    inputs = data.inputs(packet)
    require(packet_receipt['inputs'] == {k: {'bytes': len(v), 'sha256': sha(v)} for k,v in inputs.items()}, 'view_receipt', 'exact complete training inputs')
    evaluations = {}; evaluation_parents = {}
    for population, directory in [('validation', validation), ('reserved_test', reserved)]:
        reservation, records = data.read_view(directory)
        require(all(p['entry']['split'] == population for p in records), 'evaluation_split', 'independent whole evaluation parents')
        evaluations[population] = sha(canonical(reservation)); evaluation_parents[population] = records
        for slot, raw in data.inputs(directory).items(): inputs[population+'_'+slot] = raw
    train_ids = {p['entry']['trajectory_id'] for p in parents if p['entry']['split'] == 'train'}
    val_ids = {p['entry']['trajectory_id'] for p in evaluation_parents['validation']}
    test_ids = {p['entry']['trajectory_id'] for p in evaluation_parents['reserved_test']}
    require(not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids), 'split_overlap', 'no shared simulations')
    def physical(records):
        return {sha(canonical({key:p['scene'][key] for key in ('counts_xyz','cells_per_metre','force')})) for p in records}
    training_physical = physical([p for p in parents if p['entry']['split']=='train'])
    validation_physical = physical(evaluation_parents['validation'])
    reserved_physical = physical(evaluation_parents['reserved_test'])
    require(len(training_physical)==len(train_ids) and not (training_physical & validation_physical or training_physical & reserved_physical or validation_physical & reserved_physical), 'physical_split_overlap', 'distinct actual setups, not renamed duplicates')
    require({p['entry']['trajectory_id'] for p in parents if p['entry']['split']=='validation'} == val_ids, 'validation_population', 'same validation parents')
    runtime_raw = read(runtime_manifest, 16*MIB); runtime = decode(runtime_raw)
    require(runtime['profile']=='managed_torch_python_v1' and runtime['packages']['torch']==torch.__version__ and runtime['packages']['numpy']==np.__version__, 'runtime', 'qualified installed Torch release')
    snapshot = sha(runtime_raw)
    summary = {name: runtime[name] for name in ('profile','packages','python_version')}; summary['manifest_sha256'] = snapshot
    model.configure_cpu(); torch.manual_seed(config['training']['seed'])
    with patch.object(torch.optim.Optimizer, '__init__', side_effect=AssertionError('qualification must not optimize')):
        network = model.PressureCorrectionNet(); normalization = model.fit_normalization(parents,data.fit_normalization)
        checked = []
        with torch.no_grad():
            for parent in parents:
                sample = data.sample(parent, 0, labels=False)
                predicted = model.predict_pressure(network, sample, normalization)
                require(predicted.shape == sample['base'].shape and bool(torch.isfinite(predicted).all()), 'model_shape', 'same physical shape')
                sample['native_input_residual'].zero_()
                require(torch.equal(model.predict_pressure(network, sample, normalization), sample['base']), 'zero_residual', 'exact bypass')
                checked.append(parent['counts_xyz'])
        state = io.BytesIO(); np.savez(state, **{name: tensor.detach().numpy() for name,tensor in network.state_dict().items()}); state_bytes = len(state.getvalue())
    projection = (len(config['training']['checkpoints'])+1)*state_bytes + len(inputs['splits']) + 5*MIB
    require(state_bytes <= 2*MIB and projection <= config['resources']['max_output_bytes'], 'output_projection', 'bounded declared complete states')
    qualification = {'status':'passed','optimizer_constructed':False,'optimizer_updates':0,'normalization_sha256':sha(canonical(normalization)),
        'grid_shapes_xyz':checked,'zero_residual_exact':True,'physical_cells':'cubic','train_parents':len(train_ids),'validation_parents':len(val_ids),'reserved_test_parents':len(test_ids),'output_projection_bytes':projection}
    sources = {name: read(HERE/name, 2*MIB) for name in ('policy.py','support.py','scenes.py','model.py','data.py','train.py','prepared.py','physical_residual.py','analytical.py','test_inference.py')}
    for name,path in [('portable_model.py','experiments/cuda-pressure-v1/model.py'),('portable_policy.py','experiments/cuda-pressure-v1/policy.py'),
        ('frozen_gn.py','experiments/global-pressure-v1/groupnorm-v1/model.py'),('historical-reservation.json','docs/plans/global-pressure-correction-v1/reservation.json')]: sources[name] = read(LAB/path, MIB)
    for name in ('model','train','data'): sources['original_'+name+'.py'] = read(LAB/'experiments/global-pressure-v1/training'/(name+'.py'), MIB)
    sources['custody-validator.py'] = read(tuldok/'recipe_process.py', MIB)
    sources.update({'config.json':canonical(config),'runtime-manifest.json':canonical(summary),'qualification.json':canonical(qualification)})
    witness = {'schema':'rheon-water-learning-source-v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=LAB,text=True).strip(),
        'architecture':model.ARCHITECTURE,'packet_experiment_sha256':sha(canonical(packet_config)),'evaluation_reservations':evaluations,
        'experiment_sha256':sha(sources['config.json']),'runtime_manifest_sha256':sha(sources['runtime-manifest.json']),
        'sources':{name:sha(raw) for name,raw in sources.items() if name!='runtime-manifest.json'},
        'inputs':{slot:{'sha256':sha(raw),'bytes':len(raw)} for slot,raw in inputs.items()},'normalization_sha256':sha(canonical(normalization)),
        'custody_validator_sha256':sha(sources['custody-validator.py'])}
    sources['source-witness.json'] = canonical(witness)
    output = fresh(output); prefix = output.relative_to(LAB).as_posix()+'/'
    for name,raw in sources.items(): (output/name).write_bytes(raw)
    (output/'inputs').mkdir()
    input_names = {slot:slot+('.json' if slot.endswith('splits') else '.npz') for slot in inputs}
    for slot,raw in inputs.items(): (output/'inputs'/input_names[slot]).write_bytes(raw)
    sys.path.insert(0,str(tuldok))
    from recipe_contract import decode_definition
    from projects import manifest as validate_project
    recipes=[]
    for mode in ('train','restore'):
        raw=canonical(definition(config,sources,snapshot,mode,prefix));decode_definition(raw);(output/(mode+'-recipe.json')).write_bytes(raw)
        recipes.append({'id':mode,'label':mode,'path':prefix+mode+'-recipe.json','sha256':sha(raw)})
    import test_inference
    for population in ('validation','reserved_test'):
        raw=canonical(test_inference.definition(config,sources,snapshot,prefix,population));decode_definition(raw);(output/(population+'-recipe.json')).write_bytes(raw)
        recipes.append({'id':population,'label':population+' · CPU','path':prefix+population+'-recipe.json','sha256':sha(raw)})
    manifest={'schema_version':1,'id':uuid.uuid4().hex,'name':config['id'],'activity_path':'runs/live-project/activity.json',
        'data':[{'id':slot,'label':slot,'path':prefix+'inputs/'+input_names[slot],'sha256':sha(raw)} for slot,raw in inputs.items()],
        'recipes':recipes,'evidence':[],'requirements':[]}
    validate_project(manifest);write(output/'project.json',manifest)
    stages=[{'name':'train','recipe_id':'train','inputs':{slot:slot for slot in train.DATA}},
        {'name':'restore','recipe_id':'restore','inputs':{**{slot:slot for slot in train.DATA},**{slot:{'stage':'train','artifact':name} for slot,(name,_role,_cap) in train.trained_inputs(config).items()}}}]
    for population in ('validation','reserved_test'):
        stages.append({'name':population,'recipe_id':population,'inputs':{**{population+'_'+slot:population+'_'+slot for slot in train.DATA},
            **{slot:{'stage':'train','artifact':name} for slot,(name,_role,_cap) in test_inference.TRAINED.items()}}})
    write(output/'stages.json',stages)
    return {'project':str(output/'project.json'),'qualification':qualification,'source_witness_sha256':sha(sources['source-witness.json'])}
