"""Freeze the water model into captured CUDA training and CPU inference recipes."""
import io
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch

import numpy as np
import torch

from policy import HERE, LAB, MIB, canonical, decode, experiment, fresh, read, require, sha, write
import data
import model
import train


def definition(config, sources, snapshot, mode, prefix):
    inputs = {'dataset': {'max_bytes': 32*MIB}, 'splits': {'max_bytes': 4*MIB}}
    arguments = [{'literal': mode}]
    for slot in inputs:
        arguments += [{'literal': '--'+slot}, {'input': slot}]
    for name in ('config', 'runtime-manifest', 'source-witness'):
        arguments += [{'literal': '--'+name}, {'source': prefix+name+'.json'}]
    arguments += [{'literal': '--input-evidence'}, {'input_evidence': True}, {'literal': '--output-dir'}, {'output_dir': True}]
    if mode == 'restore':
        for slot, (_name, _role, cap) in train.trained_inputs(config).items():
            inputs[slot] = {'max_bytes': cap}
            arguments += [{'literal': '--'+slot.replace('_', '-')}, {'input': slot}]
    return {'schema_version': 1, 'format': 'external_recipe_v1', 'id': config['id']+'-'+mode,
            'name': 'CUDA training' if mode == 'train' else 'CPU deployment restore', 'version': 1, 'runtime': 'managed_torch_python_v1', 'runtime_device': 'cuda' if mode == 'train' else 'cpu',
            'runtime_snapshot_sha256': snapshot, 'sources': [{'path': prefix+name, 'sha256': sha(raw)} for name, raw in sorted(sources.items())],
            'entrypoint': prefix+'train.py', 'arguments': arguments, 'parameters': {}, 'inputs': inputs,
            'outputs': [{'path': name, 'role': 'evaluation_result' if mode == 'restore' and name == 'receipt.json' else role,
                         'max_bytes': cap, 'required': True} for name, (role, cap) in train.outputs(config).items()],
            'environment': {}, 'resources': config['resources']}


def assemble(packet, runtime_manifest, tuldok, output, configuration, test_packet):
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=LAB), 'source_commit', 'commit qualified sources before freezing material')
    packet_config, parents = data.load(read(packet/'dataset.npz', 32*MIB), decode(read(packet/'splits.json', 4*MIB)))
    config = experiment(decode(read(configuration, MIB)))
    receipt = decode(read(packet/'dataset-receipt.json', MIB))
    inputs = {slot: read(packet/name, 32*MIB if slot == 'dataset' else 4*MIB) for slot, name in train.DATA.items()}
    require(receipt['status'] == 'verified_lossless_water' and receipt['reader_source_sha256'] == sha(read(HERE/'data.py'))
            and receipt['dataset_sha256'] == sha(inputs['dataset']) and receipt['manifest_sha256'] == sha(inputs['splits']), 'packet_receipt', 'qualified current lossless reader')
    test_inputs = {'test_dataset': read(test_packet/'dataset.npz',32*MIB), 'test_splits': read(test_packet/'splits.json',4*MIB)}
    test_reservation, test_parents = data.load(test_inputs['test_dataset'],decode(test_inputs['test_splits']))
    require(canonical(test_reservation) == canonical(packet_config) and all(p['entry']['split']=='reserved_test' for p in test_parents), 'test_population', 'same pre-reserved independent test parents')
    inputs.update(test_inputs)
    runtime_raw = read(runtime_manifest, 16*MIB); runtime = decode(runtime_raw)
    require(runtime['profile']=='managed_torch_python_v1' and runtime['packages']['torch']==torch.__version__
            and runtime['packages']['numpy']==np.__version__, 'runtime', 'selected installed Torch release')
    runtime['manifest_sha256'] = sha(runtime_raw)
    runtime = {name:runtime[name] for name in ('profile','manifest_sha256','packages','python_version')}
    runtime_raw = canonical(runtime)
    model.configure_cpu(); torch.manual_seed(config['training']['seed'])
    with patch.object(torch.optim.Optimizer, '__init__', side_effect=AssertionError('preflight cannot create an optimizer')):
        network = model.PressureCorrectionNet(); normalization = data.fit_normalization(parents)
        checked = []
        with torch.no_grad():
            for parent in parents:
                sample = data.sample(parent, 0, labels=False)
                predicted = model.predict_pressure(network, sample, normalization)
                require(predicted.shape == sample['base'].shape and bool(torch.isfinite(predicted).all()), 'model_shape', 'same physical grid')
                sample['native_input_residual'].zero_()
                require(torch.equal(model.predict_pressure(network, sample, normalization), sample['base']), 'zero_residual', 'exact pressure bypass')
                checked.append(parent['counts_xyz'])
        state = io.BytesIO(); np.savez(state, **{name: tensor.detach().numpy() for name, tensor in network.state_dict().items()})
        state_bytes = len(state.getvalue())
    checkpoints = len(config['training']['checkpoints'])
    projection = (checkpoints+1)*state_bytes+len(inputs['splits'])+5*MIB
    require(state_bytes <= 2*MIB and projection <= config['resources']['max_output_bytes'], 'output_projection', 'complete history and publication reserve')
    qualification = {'status': 'passed', 'optimizer_constructed': False, 'optimizer_updates': 0,
                     'reserved_test_parents': len(test_parents), 'grid_shapes_xyz': checked, 'zero_residual_exact': True, 'physical_cells': 'cubic',
                     'normalization_sha256': sha(canonical(normalization)), 'output_projection_bytes': projection}
    sources = {name: read(HERE/name, 2*MIB) for name in ('policy.py', 'support.py', 'scenes.py', 'model.py', 'data.py', 'train.py', 'test_inference.py')}
    sources['portable_model.py'] = read(LAB/'experiments/cuda-pressure-v1/model.py', MIB)
    sources['portable_policy.py'] = read(LAB/'experiments/cuda-pressure-v1/policy.py', MIB)
    sources['custody-validator.py'] = read(tuldok/'recipe_process.py', MIB)
    sources['frozen_gn.py'] = read(LAB/'experiments/global-pressure-v1/groupnorm-v1/model.py', MIB)
    for name in ('model', 'train', 'data'):
        sources['original_'+name+'.py'] = read(LAB/'experiments/global-pressure-v1/training'/(name+'.py'), MIB)
    sources['historical-reservation.json'] = read(LAB/'docs/plans/global-pressure-correction-v1/reservation.json', MIB)
    sources.update({'config.json': canonical(config), 'runtime-manifest.json': runtime_raw, 'qualification.json': canonical(qualification)})
    witness = {'schema': 'rheon-sealed-water-source-v1', 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LAB, text=True).strip(),
               'architecture': model.ARCHITECTURE, 'packet_experiment_sha256': sha(canonical(packet_config)), 'experiment_sha256': sha(sources['config.json']), 'runtime_manifest_sha256': sha(runtime_raw),
               'sources': {name: sha(raw) for name, raw in sources.items() if name != 'runtime-manifest.json'},
               'inputs': {slot: {'sha256': sha(raw), 'bytes': len(raw)} for slot, raw in inputs.items()},
               'normalization_sha256': sha(canonical(normalization)), 'custody_validator_sha256': sha(read(tuldok/'recipe_process.py', MIB))}
    sources['source-witness.json'] = canonical(witness)
    output = fresh(output); prefix = output.relative_to(LAB).as_posix()+'/'
    for name, raw in sources.items():
        (output/name).write_bytes(raw)
    (output/'inputs').mkdir()
    for slot, raw in inputs.items():
        name = train.DATA[slot] if slot in train.DATA else slot+'.'+('npz' if slot=='test_dataset' else 'json')
        (output/'inputs'/name).write_bytes(raw)
    sys.path.insert(0, str(tuldok))
    from recipe_contract import decode_definition
    from projects import manifest as validate_project
    recipes = []
    for mode in ('train', 'restore'):
        raw = canonical(definition(config, sources, runtime['manifest_sha256'], mode, prefix)); decode_definition(raw)
        (output/(mode+'-recipe.json')).write_bytes(raw)
        recipes.append({'id': mode, 'label': 'CUDA training' if mode == 'train' else 'CPU deployment restore', 'path': prefix+mode+'-recipe.json', 'sha256': sha(raw)})
    manifest = {'schema_version': 1, 'id': uuid.uuid4().hex, 'name': config['id'], 'activity_path': 'runs/live-project/activity.json',
                'data': [{'id': slot, 'label': 'Qualified variable-grid '+slot, 'path': prefix+'inputs/'+(train.DATA[slot] if slot in train.DATA else slot+'.'+('npz' if slot=='test_dataset' else 'json')), 'sha256': sha(raw)} for slot, raw in inputs.items()],
                'recipes': recipes, 'evidence': [], 'requirements': []}
    import test_inference
    raw = canonical(test_inference.definition(config,sources,runtime['manifest_sha256'],prefix))
    decode_definition(raw);(output/'test-recipe.json').write_bytes(raw)
    manifest['recipes'].append({'id':'test','label':'Reserved water test · CPU inference','path':prefix+'test-recipe.json','sha256':sha(raw)})
    validate_project(manifest); write(output/'project.json', manifest)
    write(output/'stages.json', [{'name': 'train', 'recipe_id': 'train', 'inputs': {slot: slot for slot in train.DATA}},
          {'name': 'restore', 'recipe_id': 'restore', 'inputs': {**{slot: slot for slot in train.DATA}, **{slot: {'stage': 'train', 'artifact': name} for slot, (name, _role, _cap) in train.trained_inputs(config).items()}}},
          {'name':'test','recipe_id':'test','inputs':{'test_dataset':'test_dataset','test_splits':'test_splits',
            **{slot:{'stage':'train','artifact':name} for slot,(name,_role,_cap) in test_inference.TRAINED.items()}}}])
    return {'project': str(output/'project.json'), 'qualification': qualification, 'source_witness_sha256': sha(sources['source-witness.json'])}
