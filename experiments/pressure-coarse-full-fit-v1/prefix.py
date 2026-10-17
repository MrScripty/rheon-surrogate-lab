"""Bind a completed CPU prefix and exact classifier qualification for continuation."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def qualified_prefix(args,study):
    """Reuse only completed exact prefix evidence after a classified failed pool."""
    supplied=(args.qualified_prefix,args.campaign_entry,args.classification_qualification)
    if not any(supplied):return None
    if not all(supplied):raise ValueError('Explicit prefix, entry and classification qualification required together')
    prefix=args.qualified_prefix.resolve(strict=True);proof=read(args.classification_qualification)
    source=HERE.parent/'pressure-throughput-feedback-v1'
    required={str(path)for path in(HERE/'campaign.py',HERE/'rejection.py',HERE/'qualify_rejection.py',HERE/'process_ownership.py',
        source/'benchmark.py',source/'episode.py',source/'failure.py',args.runtime,prefix/'status.json',prefix/'phase-clocks.json',
        prefix/'regression/receipt.json',prefix/'regression-config.json',prefix/'timing-config.json',prefix/'first-frame/episode.json',
        prefix/'transfer.json',prefix/'common-rhs/receipt.json',prefix/'package/manifest.json')}
    if not required.issubset(proof['sources']):raise ValueError('Complete qualified prefix/source inventory required')
    if not(prefix.is_relative_to(args.root.resolve())and proof['status']=='passed'
        and proof['first_frame_numerical_output_exact']and proof['child_custody']['echild']
        and Path(args.campaign_entry).resolve()==HERE/'campaign.py'
        and all(digest(Path(path))==sha for path,sha in proof['sources'].items())):
        raise ValueError('Exact current classifier qualification required')
    state=read(prefix/'status.json');phases=read(prefix/'phase-clocks.json');campaign=read(prefix/'regression/receipt.json')
    if not(state['status']=='failed'and state['phase']=='regression'and state['weights_sha256']==study['weights_sha256']
        and state['training_run_id']==study['training_run_id']and state['restore_run_id']==study['restore_run_id']
        and [r['phase']for r in phases]==['package','first-frame','transfer','common-rhs','regression']
        and [r['returncode']for r in phases]==[0,0,0,0,1]
        and all(r['child_custody']['echild']and not r['child_custody']['signals_sent']for r in phases)
        and campaign['status']=='failed'and campaign['all_worker_children_reaped']):
        raise ValueError('Exact completed prefix and failed/drained campaign required')
    manifest=prefix/'package/manifest.json';package=read(manifest)
    for name,identity in package['files'].items():
        path=manifest.parent/name
        if path.stat().st_size!=identity['bytes']or digest(path)!=identity['sha256']:raise ValueError('Prefix package changed')
    if package['files']['weights.npz']['sha256']!=study['weights_sha256']or package['files']['installed-runtime.json']['sha256']!=digest(args.runtime):
        raise ValueError('Prefix package differs from measured Study/runtime')
    fixture=read(prefix/'first-frame/episode.json');common=read(prefix/'common-rhs/receipt.json');transfer=read(prefix/'transfer.json')
    if not(fixture['status']=='passed'and fixture['native_first_frame_inputs_exact']and fixture['manifest_sha256']==digest(manifest)
        and common['status']==transfer['status']=='passed'and common['all_workers_reaped']and len(common['cases'])==144
        and common['weights_sha256']==transfer['weights_sha256']==study['weights_sha256']
        and common['optimizer_updates']==transfer['optimizer_updates']==0):raise ValueError('Completed unchanged diagnostic prefix required')
    if digest(prefix/'regression-config.json')!=campaign['config_sha256']or digest(args.runtime)!=campaign['runtime_manifest_sha256']:
        raise ValueError('Failed campaign config/runtime changed')
    for row in phases[:4]:
        for path,sha in row['sources'].items():
            if Path(path).resolve()==HERE/'cpu_qualify.py':continue  # exact prior fixture requalified in proof
            if digest(Path(path))!=sha:raise ValueError('Completed prefix producer source changed')
    # These successful producer receipts contain their exact source inventories.
    for path,sha in common['sources'].items():
        if digest(Path(path))!=sha:raise ValueError('Common-RHS producer changed')
    for path,sha in transfer['sources'].items():
        if digest(Path(path))!=sha:raise ValueError('Transfer producer changed')
    for path,sha in fixture['active_sources'].items():
        if digest(Path(path))!=sha:raise ValueError('First-frame producer changed')
    return dict(prefix=prefix,manifest=manifest,qualification_sha256=digest(args.classification_qualification))

