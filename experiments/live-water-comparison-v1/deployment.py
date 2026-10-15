"""Export an immutable desktop CPU deployment from qualified Tuldok artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def context_policy(project, receipt):
    """Bind context selection to the exact training split artifact, not mutable input."""
    path = project / 'inputs/splits.json'
    expected = receipt['closure']['inputs']['splits']
    if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
        raise ValueError('Training context reservation drift')
    packet = json.loads(path.read_text())
    indices = packet['reservation']['configuration']['snapshot_steps']
    if (type(indices) is not list or not 1 <= len(indices) <= 512
            or any(type(index) is not int or not 0 <= index < 512 for index in indices)):
        raise ValueError('Bounded training snapshot indices required')
    return {'mode': 'native_prefix', 'minimum_native_steps': min(indices),
            'training_snapshot_indices': indices}

def export(project,artifacts,run,runtime,output):
    if output.exists():raise ValueError('Choose a fresh deployment directory')
    witness=json.loads((project/'source-witness.json').read_text())
    receipt=json.loads((artifacts/'receipt.json').read_text());history=json.loads((artifacts/'history.json').read_text());owner=json.loads(run.read_text())
    if owner['state']!='completed' or owner['custody_phase']!='drained' or receipt['mode']!='train':raise ValueError('Completed/drained training Run required')
    if receipt['closure']['source_witness_sha256']!=digest(project/'source-witness.json'):raise ValueError('Training source witness mismatch')
    selected={a['name']:a for a in owner['artifacts']}
    names=('weights.npz','normalization.json','history.json','receipt.json')
    for name in names:
        path=artifacts/name
        if digest(path)!=selected[name]['sha256'] or path.stat().st_size!=selected[name]['bytes']:raise ValueError('Run artifact mismatch: '+name)
    final=f'checkpoint-{receipt["optimizer_updates"]:03d}.npz'
    if digest(artifacts/'weights.npz')!=history['checkpoints'][final]['sha256']:raise ValueError('Final declared candidate required')
    context = context_policy(project, receipt)
    output.mkdir(parents=True);(output/'sources').mkdir()
    sources=(*witness['sources'],'runtime-manifest.json','source-witness.json')
    for name in sources:
        raw=(project/name).read_bytes()
        if name in witness['sources'] and hashlib.sha256(raw).hexdigest()!=witness['sources'][name]:raise ValueError('Source drift: '+name)
        (output/'sources'/name).write_bytes(raw)
    summary=json.loads((project/'runtime-manifest.json').read_text())
    if digest(runtime)!=summary['manifest_sha256']:raise ValueError('Registered runtime manifest mismatch')
    for name in names:shutil.copyfile(artifacts/name,output/name)
    shutil.copyfile(runtime,output/'installed-runtime.json');shutil.copyfile(HERE/'cpu_bridge.py',output/'cpu_bridge.py')
    manifest=json.loads(runtime.read_text())
    files={p.relative_to(output).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}
    descriptor={'schema':'rheon-pressure-cpu-deployment-v2','python':manifest['python'],'bridge':'cpu_bridge.py',
        'python_sha256':digest(Path(manifest['python'])),'files':files,'checkpoint_sha256':digest(output/'weights.npz'),
        'normalization_sha256':digest(output/'normalization.json'),'source_commit':witness['source_commit'],
        'training_run_id':owner['id'],'training_updates':receipt['optimizer_updates'],'profile':'sealed-water-seven-channel-float32',
        'context_policy':context,
        'physical_contract':{'density_kg_m3':1000.0,'dynamic_viscosity_pa_s':0.001,'cells':'cubic'},
        'assumption':manifest['assumption']}
    (output/'deployment.json').write_text(json.dumps(descriptor,sort_keys=True,indent=2)+'\n')
    return {'deployment':str(output/'deployment.json'),'checkpoint_sha256':descriptor['checkpoint_sha256'],'training_run_id':owner['id']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifacts','run','runtime','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(export(a.project.absolute(),a.artifacts.absolute(),a.run.absolute(),a.runtime.absolute(),a.output.absolute())))
