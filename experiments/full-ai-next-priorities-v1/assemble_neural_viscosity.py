"""Compose exact managed neural weights; replace only243 viscosity coefficients."""
import numpy as np
import shutil
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha
from managed_fit_artifacts import digest


def run():
    output=BASE/'b2-displacement-viscosity-assembly-01'
    if output.exists():raise ValueError('Fresh neural composition required')
    registration_path=DOCS/'neural-viscosity-composition-registration-01.json';reg=read(registration_path)
    for name,value in reg['source_sha256'].items():
        if sha(HERE/name)!=value:raise ValueError('Registered composition source changed')
    parent=BASE/'b2-displacement-pair-01';component=BASE/'viscosity-calibration-pair-01'
    checks={}
    for label,path in (('parent',parent),('component',component)):
        receipt=read(path/'receipt.json');restore=read(path/'restore/receipt.json')
        if receipt['status']!='completed'or not all(receipt['artifact_parity'].values())or restore['optimizer_updates']!=0 or not restore['full_cpu_probe_bitwise_verified']:
            raise ValueError('Exact completed managed CPU parents required')
        if restore['originating_training_run']['run_id']!=receipt['training_run_id']:
            raise ValueError('CPU parent provenance differs')
        checks[label]=dict(pair_receipt_sha256=sha(path/'receipt.json'),weights_sha256=sha(path/'restore/weights.npz'),
            training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id'],optimizer_updates=receipt['optimizer_updates'])
        if checks[label]['pair_receipt_sha256']!=reg[label+'_pair_receipt_sha256']:raise ValueError('Registered managed parent changed')
    qualification=BASE/'viscosity-calibration-native-evaluation-01.json';q=read(qualification)
    if q['status']!='passed_component_qualification'or len(q['rows'])!=24 or sha(qualification)!=reg['native_component_evaluation_sha256']:
        raise ValueError('Qualified learned float32 viscosity required')
    with np.load(parent/'restore/weights.npz',allow_pickle=False)as arrays:
        original={n:arrays[n].copy()for n in arrays.files}
    with np.load(component/'restore/weights.npz',allow_pickle=False)as arrays:
        if arrays.files!=['projection.weight']:raise ValueError('Complete learned viscosity coefficient inventory required')
        replacement=arrays['projection.weight'].copy()
    target='viscous_head.weight'
    if replacement.dtype.str!='<f4'or replacement.shape!=(3,3,3,3,3)or original[target].shape!=replacement.shape:
        raise ValueError('Exact243-coefficient shape/dtype required')
    combined=dict(original);combined[target]=replacement
    changed=[n for n in original if not np.array_equal(original[n].view('u1'),combined[n].view('u1'))]
    if changed!=[target]:raise ValueError('Only the learned viscosity tensor may change')
    output.mkdir(parents=True);np.savez(output/'weights.npz',**combined)
    shutil.copyfile(parent/'restore/normalization.json',output/'normalization.json')
    inventory={n:dict(shape=list(v.shape),dtype=v.dtype.str,sha256=digest(v.tobytes()))for n,v in combined.items()}
    write(output/'receipt.json',dict(status='assembled_from_exact_managed_CPU_parents',identifier='b2-displacement-viscosity',
        parents=checks,parent_pair=str(parent.relative_to(ROOT)),component_pair=str(component.relative_to(ROOT)),
        weights_sha256=sha(output/'weights.npz'),normalization_sha256=sha(output/'normalization.json'),inventory=inventory,
        changed_tensors=changed,coefficient_parameters=243,all_other_tensors_bitwise_unchanged=True,
        registration_sha256=sha(registration_path),native_component_evaluation_sha256=sha(qualification),
        assembler_sha256=sha(__file__),optimizer_updates=0,local_assembly_is_not_a_new_training_run=True,
        numerical_viscosity_in_candidate_inference=False,reserved_scenes_consumed=False,model_promotion=False))
    print('Assembled full neural model from two exact managed parents; only243 coefficients replaced',flush=True)


if __name__=='__main__':run()
