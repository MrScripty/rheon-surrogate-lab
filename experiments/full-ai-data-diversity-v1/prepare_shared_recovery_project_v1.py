"""Capture drained failed-Run checkpoints and a bounded32cube recovery probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

from support import ROOT, read, write, digest
from captured_long_training_bank_v1 import CapturedLongTrainingBank
from managed_input_stream_v1 import streamed_identity


def run(args):
    terminal_path = ROOT / 'runs/full-ai-data-diversity-v1/shared-temporal-pair-01/terminal-fe69c8cbcbe14f3b974ea70c1d5acc0e.json'
    terminal = read(terminal_path)
    if terminal['state'] != 'failed' or terminal['custody_phase'] != 'drained':
        raise ValueError('Original Run must be authoritatively failed and drained')
    staging = Path('/tmp/rheon-shared-long-training-candidate-01')
    old = staging / 'project-01'
    project = staging / 'project-recovery-01'
    if project.exists():
        raise ValueError('Fresh recovery Project required')
    project.mkdir()
    failed = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres/workspace-runs/.external-fe69c8cbcbe14f3b974ea70c1d5acc0e')
    scores = {'4': [], '8': []}
    log = (failed / 'custody/stdout.log').read_text()
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for width in ('4', '8'):
            if row.get('format') == 'tuldok_progress_v1' and row.get('stage') == 'width' + width + '-selection':
                if row['completed'] != len(scores[width]) + 1:
                    raise ValueError('Original epoch-score sequence is incomplete')
                scores[width].append(row['metrics']['loss'])
    if any(len(v) != 4 for v in scores.values()):
        raise ValueError('Both full4epoch fits must have completed')
    checkpoint_slots = dict(original_weights='weights.npz', original_initial='initial-weights.npz', original_last='last-evaluated-weights.npz')
    specs = {}
    for slot, name in checkpoint_slots.items():
        source = failed / 'output' / name
        before = streamed_identity(source)
        shutil.copyfile(source, project / name)
        if streamed_identity(project / name) != before:
            raise ValueError('Preserved checkpoint copy differs')
        specs[slot] = before
    bank_info = read(ROOT / 'runs/full-ai-data-diversity-v1/long-training-transport-01/results.json')['archive']
    bank = CapturedLongTrainingBank(bank_info['path'], {n: bank_info[n] for n in ('bytes', 'sha256')})
    try:
        identifier = 'long-v1-continuous-stationary-r0-cpm8'
        scene = bank.manifest['masters'][identifier]['scene']
        if scene['counts'] != [32, 32, 32]:
            raise ValueError('Explicit bounded probe scene layout required')
        arrays = bank.load(identifier, 0)
        np.savez(project / 'probe-state.npz', **{n: arrays[n] for n in ('pressure', 'vx', 'vy', 'vz', 'tracer')})
    finally:
        bank.close()
    old_meta = read(old / 'manifest.json')
    expected_bytes = 4 * 4 * (2*32**3 + 3*33*32**2) * 8 + 128
    if expected_bytes > 24*1024**2:
        raise ValueError('Recovery probe preflight bound exceeded')
    meta = dict(original_terminal=terminal, original_terminal_sha256=digest(terminal_path),
        original_checkpoint_identities=specs, original_stdout_sha256=hashlib.sha256(log.encode()).hexdigest(),
        epoch_scores=scores, config=read(old / 'config.json'), source_manifest=old_meta['source_manifest'],
        probe_scene=scene, probe_start_fine_step=0, probe_native_identifier=identifier,
        maximum_probe_bytes=24*1024**2, expected_probe_file_bytes=expected_bytes,
        original_wrong_probe_grid=[64,64,64], original_wrong_probe_expected_bytes=84672640,
        forbidden_repository_root=str(ROOT), new_parameter_fits=0, optimizer_updates=0)
    write(project / 'manifest.json', meta)
    shutil.copyfile(failed / 'custody/stdout.log', project / 'original-training-stdout.log')
    for name in ('model-source.zip', 'normalization.json', 'runtime-manifest.json', 'portable_shared_runtime_v1.py',
                 'managed_input_stream_v1.py', 'managed_fit_artifacts.py', 'custody_validator.py'):
        shutil.copyfile(old / name, project / name)
    shutil.copyfile(Path(__file__).with_name('recover_shared_temporal_pair_v1.py'), project / 'recover_shared_temporal_pair_v1.py')
    prefix = 'project-recovery-01/'
    sources = [dict(path=prefix + p.name, sha256=digest(p)) for p in project.glob('*.py')]
    static = dict(manifest=project/'manifest.json', normalization=project/'normalization.json',
        runtime_manifest=project/'runtime-manifest.json', model_source=project/'model-source.zip',
        probe_state=project/'probe-state.npz', **{slot:project/name for slot,name in checkpoint_slots.items()})
    names = [('weights.npz','checkpoint',128*1024),('initial-weights.npz','checkpoint',128*1024),
        ('last-evaluated-weights.npz','checkpoint',128*1024),('history.json','training_diagnostics',8*1024**2),
        ('probe.npy','predictions',24*1024**2),('normalization.json','preprocessing',1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map = dict(trained_weights='weights.npz',trained_initial='initial-weights.npz',trained_last='last-evaluated-weights.npz',
        trained_history='history.json',trained_probe='probe.npy',trained_receipt='receipt.json')
    recipes=[]
    for phase in ('recover','restore'):
        slots=list(static) if phase=='recover' else ['manifest','normalization','runtime_manifest','model_source']
        inputs={n:dict(max_bytes=32*1024**2) for n in slots}
        arguments=[dict(literal=phase)]
        for name in slots:
            arguments.extend([dict(literal='--'+name.replace('_','-')),dict(input=name)])
        if phase=='restore':
            for slot,name in restore_map.items():
                inputs[slot]=dict(max_bytes=next(cap for n,role,cap in names if n==name))
                arguments.extend([dict(literal='--'+slot.replace('_','-')),dict(input=slot)])
        arguments.extend([dict(literal='--input-evidence'),dict(input_evidence=True),dict(literal='--output-dir'),dict(output_dir=True)])
        definition=dict(schema_version=1,format='external_recipe_v1',id='shared-temporal-packaging-'+phase,name='Shared temporal zero-update checkpoint '+phase,version=1,
            runtime='managed_torch_python_v1',runtime_snapshot_sha256=digest(static['runtime_manifest']),runtime_device='cuda' if phase=='recover' else 'cpu',
            sources=sources,entrypoint=prefix+'recover_shared_temporal_pair_v1.py',arguments=arguments,parameters={},inputs=inputs,
            outputs=[dict(path=n,role='evaluation_result' if phase=='restore' and n=='receipt.json' else role,max_bytes=cap,required=True) for n,role,cap in names],
            environment={},resources=dict(wall_seconds=600 if phase=='recover' else 300,max_output_bytes=64*1024**2,cpu_cores=1,memory_bytes=2*1024**3))
        write(project/(phase+'-recipe.json'),definition)
        recipes.append(dict(id=phase,label=phase,path=prefix+phase+'-recipe.json',sha256=digest(project/(phase+'-recipe.json'))))
    document=dict(schema_version=2,id=uuid.uuid4().hex,name='Recover completed dualwidth fits without additional optimization',
        data=[dict(id=n,label=n,path=str(p.relative_to(staging)),sha256=digest(p)) for n,p in static.items()],recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(project/'project.json',document)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok-large-input-v3'))
    import projects,recipe_contract
    projects.manifest(document)
    for phase in ('recover','restore'):recipe_contract.decode_definition((project/(phase+'-recipe.json')).read_bytes())
    write(project/'assembly.json',dict(project_root=str(staging),project_root_alias='long-training',project=prefix+'project.json',project_sha256=digest(project/'project.json'),
        sources=sources,static_inputs={n:dict(path=str(p),**streamed_identity(p)) for n,p in static.items()},restore_artifact_map=restore_map,
        original_training_run=terminal['id'],new_parameter_fits=0,recovered_fits=2,expected_probe_file_bytes=expected_bytes))
    print('Prepared zero-update recovery from drained failed Run; exact original checkpoints retained; probe bound checked.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);run(p.parse_args())
