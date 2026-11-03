"""Read-only final reconciliation of fitted tensors, populations, Runs and workers."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/pressure-methods-v3'
DOCS=ROOT/'docs/plans/pressure-methods-v3'
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
    str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1'),str(ROOT/'runs/dependency-migration-v1/Tuldok')]
from common import read,write,sha

def run():
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    destination=BASE/'managed-audit-01'
    if destination.exists():raise ValueError('Fresh final reconciliation required')
    destination.mkdir();runs=[];fits=[];evidence={}
    def bind(path):evidence[str(path.relative_to(ROOT))]=sha(path)
    small=[]
    for arm in ['v3-replay','v3-native','v3-c37half','v3-c44half']:
        generation=3 if arm=='v3-replay'else 7
        study=BASE/f'{arm}-study-{generation:02d}';receipt=read(study/'receipt.json')
        training=read(study/'train/receipt.json');restore=read(study/'restore/receipt.json')
        qualification=read(BASE/f'{arm}-qualification-{generation:02d}.json')
        entry=read(BASE/'entries'/f'{arm}.json');project=ROOT/entry['project'];witness=read(project/'source-witness.json')
        if sha(project/'project.json')!=qualification['project_sha256']or sha(project/'source-witness.json')!=qualification['source_witness_sha256']:
            raise ValueError('Qualified final Project identity changed')
        for name,expected in witness['sources'].items():
            if sha(project/name)!=expected:raise ValueError('Final captured model source changed: '+name)
        for path,expected in entry['identities'].items():
            if sha(ROOT/path)!=expected:raise ValueError('Restored evaluation entry changed')
        if receipt['status']!='completed'or not all(receipt['artifact_parity'].values()):raise ValueError('Managed fit/restore parity incomplete')
        if training['optimizer_updates']!=2400 or training['execution']['device']!='cuda:0':raise ValueError('Actual CUDA optimizer budget differs')
        if restore['optimizer_updates']!=0 or restore['execution']['cuda_initialized']:raise ValueError('Fresh CPU restore contract differs')
        if qualification['small_replay']['matching_updates']!=1600 or not qualification['small_replay']['exact_sample_subsequence']:
            raise ValueError('Small replay guarantee missing')
        small.append(qualification['sampling']['small_subsequence_sha256'])
        if 'half'in arm and qualification['parameters']!=92560:raise ValueError('Reduced-width tensor count differs')
        if arm!='v3-replay':
            if qualification['largest_counts_xyz']!=[128]*3 or qualification['largest_native_frames']!=[0,15]:raise ValueError('Native cold/warm qualification missing')
            if qualification['sampling']['tier_exposures']!={'16':1600,'32':267,'64':267,'128':266}:raise ValueError('Native sampler budget differs')
            if training['prediction_inventory']!=restore['prediction_inventory']or not any(v['shape']==[128]*3 for v in restore['prediction_inventory'].values()):
                raise ValueError('Fresh CPU restore lacks exact full128³ prediction parity')
            if training['outputs']['predictions.npz']['bytes']>8*1024**2 or not qualification['prediction_export_smoke']['full_resolution_cpu_restore_hash_check_retained']:
                raise ValueError('Bounded diagnostic export/full-field proof missing')
            transfer=BASE/f'{arm}-transfer-07.json'
            if read(transfer)['status']!='passed':raise ValueError('Native128³ complete predictor transfer not verified')
            bind(transfer)
        for phase,key in [('fit','training_run_id'),('cpu-restore','restore_run_id')]:
            identifier=receipt[key];remote=owner.call('runs.get',dict(run_id=identifier));path=destination/(identifier+'.json');write(path,remote)
            if remote['state']!='completed'or remote['custody_phase']!='drained':raise ValueError('Run is not completed and drained')
            runs.append(dict(arm=arm,phase=phase,run_id=identifier,state=remote['state'],custody_phase=remote['custody_phase']));bind(path)
        for path in [study/'receipt.json',study/'train/receipt.json',study/'restore/receipt.json',BASE/f'{arm}-qualification-{generation:02d}.json']:bind(path)
        result=read(BASE/'evaluation'/f'{arm}.json')
        old=[r for r in result['rows']if r['population']!='native-development']
        if len(old)!=156:raise ValueError('Common prior development coverage incomplete')
        fits.append(dict(arm=arm,parameters=qualification['parameters'],artifact_parity=receipt['artifact_parity']))
    if len(set(small))!=1 or len({r['run_id']for r in runs})!=8:raise ValueError('Unpaired replay or duplicate Run identities')
    failed=owner.call('runs.get',dict(run_id='ea2b0852c42e41e49b07985307a914de'))
    progress=owner.call('runs.progress',dict(run_id='ea2b0852c42e41e49b07985307a914de'))
    if failed['state']!='failed'or failed['custody_phase']!='drained'or progress['progress']['completed']!=2400:
        raise ValueError('Failed publication attempt not reconciled')
    write(destination/'failed-publication.json',failed);write(destination/'failed-publication-progress.json',progress)
    bind(destination/'failed-publication.json');bind(destination/'failed-publication-progress.json')
    interrupted=owner.call('runs.get',dict(run_id='1e1c146e73084356b6f57687992512d1'))
    interrupted_progress=owner.call('runs.progress',dict(run_id=interrupted['id']))
    reconciliation=read(BASE/'native-06-custody-reconciliation.json')
    if interrupted['state']!='interrupted'or interrupted_progress['evidence']!='custody_verified'or interrupted_progress['progress']['completed']!=2400 or not reconciliation['no_live_run_processes']:
        raise ValueError('Interrupted publication attempt not reconciled')
    write(destination/'interrupted-publication.json',interrupted);write(destination/'interrupted-publication-progress.json',interrupted_progress)
    bind(destination/'interrupted-publication.json');bind(destination/'interrupted-publication-progress.json');bind(BASE/'native-06-custody-reconciliation.json')
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json');original=[]
    for identifier in ['C37','C44','C53']:
        entry=next(m for m in material['models']if m['id']==identifier)
        for path,expected in entry['identities'].items():
            if sha(ROOT/path)!=expected:raise ValueError('Original named checkpoint changed')
        for name,expected in read(ROOT/entry['project']/'source-witness.json')['sources'].items():
            if sha(ROOT/entry['project']/name)!=expected:raise ValueError('Original captured source changed')
        original.append(dict(identifier=identifier,unchanged=True,weights_sha256=entry['weights_sha256']))
    identifiers=['C37','C44','C53','v2-headshape','v2-num6','v3-replay','v3-native','v3-c37half','v3-c44half']
    for identifier in identifiers:
        path=BASE/'evaluation'/f'{identifier}.json';result=read(path)
        for passes in [8,16]:
            if len([r for r in result['rows']if r['population']=='native-development'and r['max_passes']==passes])!=8:
                raise ValueError('Reserved native development matrix incomplete')
        bind(path)
    selection=read(DOCS/'selection.json');blind_ids={r['identifier']for r in selection['selected']}|{'v2-headshape','v2-num6'}
    for identifier in blind_ids:
        path=BASE/'blind-evaluation'/f'{identifier}.json';result=read(path)
        if result['status']!='completed'or len(result['rows'])!=8 or any(r['max_passes']!=16 for r in result['rows']):raise ValueError('Reserved blind coverage incomplete')
        bind(path)
    reservation=read(DOCS/'reservation.json');counts={}
    for row in reservation['rows']:
        path=BASE/'native-data'/row['trajectory_id']/'receipt.json';d=read(path)
        if d['status']!='completed'or d['scene']!=row['scene']or not d['native_process']['reaped']or[f['index']for f in d['snapshots']]!=[0,15]:
            raise ValueError('Native scene/snapshot/physical qualification incomplete')
        if max(c['relative_residual']for c in d['callbacks'])>1e-8:raise ValueError('Native spectral teacher is insufficiently accurate')
        if row['population']=='blind'and path.stat().st_mtime<(DOCS/'selection.json').stat().st_mtime:raise ValueError('Blind capture preceded selection')
        counts[row['population']]=counts.get(row['population'],0)+1;bind(path)
    if counts!={'train':8,'development':4,'blind':4}:raise ValueError('Whole-scene reservation count differs')
    view=read(BASE/'derived-training-view-01/manifest.json')
    if not view['exact_full_decoded_population_verified']or len(view['parents'])!=34 or view['encoded_total_bytes']>204*1024**2:
        raise ValueError('Lossless full training population guarantee missing')
    bind(BASE/'derived-training-view-01/manifest.json')
    for rep in range(3):
        a=read(BASE/f'profile-quiet-base-{rep}/receipt.json');b=read(BASE/f'profile-quiet-shared-{rep}/receipt.json')
        if any(x['arrays'][k]['sha256']!=y['arrays'][k]['sha256']for x,y in zip(a['snapshots'],b['snapshots'])for k in x['arrays']):
            raise ValueError('Optimized sampler changed native request/reference fields')
        for stem in ['base','shared']:bind(BASE/f'profile-quiet-{stem}-{rep}/receipt.json')
    a=read(BASE/'profile-parity128-base/receipt.json');b=read(BASE/'profile-parity128-shared/receipt.json')
    if any(x['arrays'][k]['sha256']!=y['arrays'][k]['sha256']for x,y in zip(a['snapshots'],b['snapshots'])for k in x['arrays']):
        raise ValueError('Optimized sampler changed full128³ native requests/reference fields')
    bind(BASE/'profile-parity128-base/receipt.json');bind(BASE/'profile-parity128-shared/receipt.json')
    for rep in range(3):
        a=read(BASE/f'profile-aligned-base-{rep}/receipt.json');b=read(BASE/f'profile-aligned-fast-{rep}/receipt.json')
        if any(x['arrays'][k]['sha256']!=y['arrays'][k]['sha256']for x,y in zip(a['snapshots'],b['snapshots'])for k in x['arrays']):
            raise ValueError('Exact-sample optimization changed native request/reference bytes')
        for stem in ['base','fast']:bind(BASE/f'profile-aligned-{stem}-{rep}/receipt.json')
    a=read(BASE/'profile-aligned128-base/receipt.json');b=read(BASE/'profile-aligned128-fast/receipt.json')
    if any(x['arrays'][k]['sha256']!=y['arrays'][k]['sha256']for x,y in zip(a['snapshots'],b['snapshots'])for k in x['arrays']):
        raise ValueError('Exact-sample optimization changed full128³ native fields')
    bind(BASE/'profile-aligned128-base/receipt.json');bind(BASE/'profile-aligned128-fast/receipt.json')
    for directory in ['quiet-all-batch-01','aligned-batch-01']:
        p=BASE/directory/'execution.json';d=read(p)
        if d['status']!='completed'or d['active']or not d['owned_child_reaped']:raise ValueError('Timing owner is not terminal')
        bind(p)
    for path in (BASE/'quiet').glob('*.json'):
        d=read(path)
        if d['status']!='completed'or d['summary']['cases']!=128 or d['cpu']!=23 or d['threads']!=1:raise ValueError('Common population timing incomplete')
        bind(path)
    if len(list((BASE/'quiet').glob('*.json')))!=11:raise ValueError('All11matched timing operating points required')
    figures=read(DOCS/'figure-evidence.json')
    for path,expected in figures['inputs'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Figure input changed')
    for artifact in figures['artifacts']:
        if sha(ROOT/artifact['path'])!=artifact['sha256']:raise ValueError('Scientific figure export changed')
        bind(ROOT/artifact['path'])
    bind(DOCS/'figure-evidence.json');bind(BASE/'quiet-start.json')
    if read(BASE/'prediction-export-tests-07.json')['status']!='passed':raise ValueError('Full-field export regressions not verified')
    bind(BASE/'prediction-export-tests-07.json')
    for directory in ['profile-build-02','coordinates-build-01','aligned-build-01']:
        p=BASE/directory/'build-receipt.json';d=read(p)
        if d['status']!='qualified_build'or sha(BASE/directory/'rheon-large-single-track')!=d['binary_sha256']:
            raise ValueError('Qualified native binary changed')
        for path,expected in d['sources'].items():
            if sha(ROOT/path)!=expected:raise ValueError('Qualified native source changed')
        bind(p)
    coupled=list((BASE/'coupled').glob('*/receipt.json'))
    if len(coupled)!=2*len(blind_ids-{'v2-num6'})+2:raise ValueError('Coupled confirmation matrix incomplete')
    for path in coupled:
        d=read(path)
        if not d['native_process']['reaped']or any(f['advanced']and not f['quality']['qualified']for f in d['frames']):
            raise ValueError('Coupled candidate custody/host gates differ')
        bind(path)
    # Read host /proc, matching executable/script arguments rather than shell text.
    live=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit()or int(path.name)==os.getpid():continue
        try:argv=[v.decode(errors='replace')for v in (path/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not argv:continue
        matched=argv[0].startswith(str(BASE))and Path(argv[0]).name=='rheon-large-single-track'
        if Path(argv[0]).name.startswith('python'):
            for argument in argv[1:]:
                if argument.endswith('.py')and'\n'not in argument:
                    candidate=(ROOT/argument).resolve()
                    if candidate.is_relative_to(ROOT/'experiments/pressure-methods-v3')or candidate.is_relative_to(BASE):matched=True
            if any(v.endswith('pressure-methods-v2/run_batch.py')for v in argv)and any(str(BASE)in v or v.startswith('runs/pressure-methods-v3/')for v in argv):matched=True
        if matched:live.append(dict(pid=int(path.name),program=argv[0]))
    if live:raise ValueError('Owned campaign workers remain live: '+str(live))
    for path in [DOCS/'selection.json',DOCS/'selection-rule.json',DOCS/'reservation.json']:bind(path)
    result=dict(status='passed',successful_fits=4,successful_managed_runs=8,managed_runs=runs,fits=fits,
        failed_post_training_publications=1,interrupted_post_training_publications=1,
        failed_publication_run_id=failed['id'],interrupted_publication_run_id=interrupted['id'],actual_optimizer_updates_including_unpublished_attempts=14400,
        original_checkpoints=original,native_trajectory_counts=counts,common_small_subsequence_sha256=small[0],
        live_owned_workers=[],host_namespace_process_audit=True,evidence=evidence,source_sha256=sha(__file__),
        promotion=False,application_drift_acceptance_claimed=False,
        scope='Measured controlled exploration, including negative results. Physical solver gates remain distinct from application accuracy and long-horizon drift.')
    write(DOCS/'final-audit.json',result)
    print('Audited four actual fits/eight drained Runs; immutable baselines, replay, blind split and terminal workers verified',flush=True)

if __name__=='__main__':run()
