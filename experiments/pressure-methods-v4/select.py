"""Freeze candidate identities/policies only after the complete development matrix."""
from datetime import datetime,timezone
import math
from v4_support import ROOT,BASE,DOCS,read,write,sha

def freeze():
    output=DOCS/'selection.json'
    if output.exists():raise ValueError('Selection is immutable; inspect existing frozen identities')
    baseline=['v3-native-'+p for p in ['learned','vcycle-neural','vcycle-only','gcr-learned','gcr-multilevel','large-channels-last','spectral']]+['v2-num6-numerical']
    students=[a+'-learned'for a in ['v4-student75-distill','v4-student75-control','v4-student50-distill','v4-student50-control']]
    full=['v4-replay-learned','v4-replay-gcr-learned','v4-gcr-gcr-learned','v4-gcr-learned','v4-diverse-learned','v4-onpolicy-learned']
    records={};evidence={};expected=None
    for key in baseline+students+full:
        path=BASE/'evaluation'/(key+'.json');d=read(path)
        if d['status']!='completed'or d['optimizer_updates']!=0:raise ValueError('Completed zero-update development matrix required')
        if d['source_sha256']!=sha(ROOT/'experiments/pressure-methods-v4/evaluate_candidate.py'):raise ValueError('Evaluator source changed')
        identities=[(r['population'],r['trajectory_id'],r['frame'])for r in d['rows']]
        if len(identities)!=174 or len(set(identities))!=174:raise ValueError('Complete unique common population required')
        if expected is None:expected=set(identities)
        if set(identities)!=expected:raise ValueError('Candidate population differs')
        for source,digest in d['dependency_sources'].items():
            if sha(ROOT/source)!=digest:raise ValueError('Solver implementation changed')
        for source,digest in d['entry']['identities'].items():
            if sha(ROOT/source)!=digest:raise ValueError('Completed checkpoint changed')
        development=[r for r in d['rows']if r['population']in ['native-development','v4-development']]
        retained=[r for r in d['rows']if r['population']not in ['native-development','v4-development']]
        if len(development)!=18 or len(retained)!=156:raise ValueError('Development/retention partition differs')
        evidence[str(path.relative_to(ROOT))]=sha(path)
        records[key]=dict(identifier=key,entry=d['entry'],policy=d['policy'],evaluation=str(path.relative_to(ROOT)),
            development_cases=18,development_qualified=sum(r['qualified']for r in development),
            worst_development_pressure_relative_l2=max(r['pressure_relative_l2']for r in development),
            retained_cases=156,retained_qualified=sum(r['qualified']for r in retained))
    fits={}
    for arm in ['v4-student75-distill','v4-student75-control','v4-student50-distill','v4-student50-control','v4-replay','v4-gcr','v4-diverse','v4-onpolicy']:
        generation=2 if arm=='v4-student75-distill'else 1
        path=BASE/(arm+f'-study-{generation:02d}')/'receipt.json';d=read(path)
        if d['status']!='completed'or not all(d['artifact_parity'].values()):raise ValueError('Every planned fit and fresh restore must complete before blind selection')
        transfer_path=BASE/'transfer'/(arm+'.json');transfer=read(transfer_path)
        if transfer['status']!='passed'or len(transfer['rows'])!=2 or transfer['entry']['weights_sha256']!=d['weights_sha256']:
            raise ValueError('Actual full128 cold/warm predictor transfer required before selection')
        evidence[str(transfer_path.relative_to(ROOT))]=sha(transfer_path)
        fits[arm]=dict(training_run_id=d['training_run_id'],restore_run_id=d['restore_run_id'],receipt_sha256=sha(path))
    def rank(key):
        r=records[key]
        return(-r['development_qualified'],r['worst_development_pressure_relative_l2'],key)
    selected=[dict(family='student',**records[min(students,key=rank)]),
        dict(family='full',**records[min(full,key=rank)]),dict(family='multilevel-hybrid',**records['v3-native-vcycle-neural'])]
    references=[records[k]for k in ['v3-native-learned','v2-num6-numerical','v3-native-spectral','v3-native-vcycle-only','v3-native-gcr-multilevel',
        'v4-replay-learned','v4-replay-gcr-learned']]
    write(output,dict(status='frozen_before_blind_capture',selected=selected,references=references,
        all_evaluations=evidence,managed_fits=fits,selection_rule_sha256=sha(DOCS/'selection-rule.json'),
        scene_reservation_sha256=sha(DOCS/'scene-reservation.json'),source_sha256=sha(__file__),frozen_utc=datetime.now(timezone.utc).isoformat(),
        no_model_promotion=True,application_acceptance=False,
        interpretation='Selection is diagnostic. Historical retention failures remain visible and do not become qualified merely through selection. Timing does not select a candidate.'))
    print('Frozen blind selection',[(r['family'],r['identifier'])for r in selected],flush=True)

if __name__=='__main__':freeze()
