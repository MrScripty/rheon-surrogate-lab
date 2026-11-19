"""Freeze the qualified variable-grid model into ordinary captured CPU recipes."""
import io
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch

import numpy as np
import torch

from policy import HERE, LAB, MIB, canonical, decode, fresh, read, require, sha, write
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
            'name': 'Variable-grid CPU '+mode, 'version': 1, 'runtime': 'captured_cpu_python_v1',
            'runtime_snapshot_sha256': snapshot, 'sources': [{'path': prefix+name, 'sha256': sha(raw)} for name, raw in sorted(sources.items())],
            'entrypoint': prefix+'train.py', 'arguments': arguments, 'parameters': {}, 'inputs': inputs,
            'outputs': [{'path': name, 'role': 'evaluation_result' if mode == 'restore' and name == 'receipt.json' else role,
                         'max_bytes': cap, 'required': True} for name, (role, cap) in train.outputs(config).items()],
            'environment': {}, 'resources': config['resources']}


def assemble(packet, runtime_manifest, tuldok, output):
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=LAB), 'source_commit', 'commit qualified sources before freezing material')
    config, parents = data.load(read(packet/'dataset.npz', 32*MIB), decode(read(packet/'splits.json', 4*MIB)))
    receipt = decode(read(packet/'dataset-receipt.json', MIB))
    inputs = {slot: read(packet/name, 32*MIB if slot == 'dataset' else 4*MIB) for slot, name in train.DATA.items()}
    require(receipt['status'] == 'verified_lossless_development' and receipt['reader_source_sha256'] == sha(read(HERE/'data.py'))
            and receipt['dataset_sha256'] == sha(inputs['dataset']) and receipt['manifest_sha256'] == sha(inputs['splits']), 'packet_receipt', 'qualified current lossless reader')
    runtime_raw = read(runtime_manifest, 16*MIB); runtime = decode(runtime_raw)
    require(runtime['profile'] == 'captured_cpu_python_v1' and runtime['packages']['torch'] == torch.__version__ == '2.8.0+cpu'
            and runtime['packages']['numpy'] == np.__version__ == '2.3.5' and torch.version.cuda is None, 'runtime', 'actual CPU snapshot')
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
    checkpoints = 1+config['training']['updates']//config['training']['checkpoint_interval']
    projection = (checkpoints+1)*state_bytes+len(inputs['splits'])+5*MIB
    require(state_bytes <= 2*MIB and projection <= config['resources']['max_output_bytes'], 'output_projection', 'complete history and publication reserve')
    qualification = {'status': 'passed', 'optimizer_constructed': False, 'optimizer_updates': 0,
                     'grid_shapes_xyz': checked, 'zero_residual_exact': True, 'physical_cells': 'cubic',
                     'normalization_sha256': sha(canonical(normalization)), 'output_projection_bytes': projection}
    sources = {name: read(HERE/name, 2*MIB) for name in ('policy.py', 'support.py', 'model.py', 'data.py', 'train.py')}
    sources['frozen_gn.py'] = read(LAB/'experiments/global-pressure-v1/groupnorm-v1/model.py', MIB)
    for name in ('model', 'train', 'data'):
        sources['original_'+name+'.py'] = read(LAB/'experiments/global-pressure-v1/training'/(name+'.py'), MIB)
    sources['historical-reservation.json'] = read(LAB/'docs/plans/global-pressure-correction-v1/reservation.json', MIB)
    sources.update({'config.json': canonical(config), 'runtime-manifest.json': runtime_raw, 'qualification.json': canonical(qualification)})
    witness = {'schema': 'rheon-variable-pressure-source-v2', 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LAB, text=True).strip(),
               'architecture': model.ARCHITECTURE, 'experiment_sha256': sha(sources['config.json']), 'runtime_manifest_sha256': sha(runtime_raw),
               'sources': {name: sha(raw) for name, raw in sources.items() if name != 'runtime-manifest.json'},
               'inputs': {slot: {'sha256': sha(raw), 'bytes': len(raw)} for slot, raw in inputs.items()},
               'normalization_sha256': sha(canonical(normalization)), 'custody_validator_sha256': sha(read(tuldok/'recipe_process.py', MIB))}
    sources['source-witness.json'] = canonical(witness)
    output = fresh(output); prefix = output.relative_to(LAB).as_posix()+'/'
    for name, raw in sources.items():
        (output/name).write_bytes(raw)
    (output/'inputs').mkdir()
    for slot, raw in inputs.items():
        (output/'inputs'/train.DATA[slot]).write_bytes(raw)
    sys.path.insert(0, str(tuldok))
    from recipe_contract import decode_definition
    from projects import manifest as validate_project
    recipes = []
    for mode in ('train', 'restore'):
        raw = canonical(definition(config, sources, sha(runtime_raw), mode, prefix)); decode_definition(raw)
        (output/(mode+'-recipe.json')).write_bytes(raw)
        recipes.append({'id': mode, 'label': 'Variable-grid CPU '+mode, 'path': prefix+mode+'-recipe.json', 'sha256': sha(raw)})
    manifest = {'schema_version': 1, 'id': uuid.uuid4().hex, 'name': config['id'], 'activity_path': 'runs/live-project/activity.json',
                'data': [{'id': slot, 'label': 'Qualified variable-grid '+slot, 'path': prefix+'inputs/'+train.DATA[slot], 'sha256': sha(raw)} for slot, raw in inputs.items()],
                'recipes': recipes, 'evidence': [], 'requirements': []}
    validate_project(manifest); write(output/'project.json', manifest)
    write(output/'stages.json', [{'name': 'train', 'recipe_id': 'train', 'inputs': {slot: slot for slot in train.DATA}},
          {'name': 'restore', 'recipe_id': 'restore', 'inputs': {**{slot: slot for slot in train.DATA}, **{slot: {'stage': 'train', 'artifact': name} for slot, (name, _role, _cap) in train.trained_inputs(config).items()}}}])
    return {'project': str(output/'project.json'), 'qualification': qualification, 'source_witness_sha256': sha(sources['source-witness.json'])}
