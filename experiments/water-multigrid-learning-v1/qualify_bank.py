"""Check declared training banks against independent cosine eigenvalue labels."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-conditioning-v1/training'))
import analytical
import data
import model
import policy
import torch


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','packet','legacy-source','legacy-config','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();config=policy.experiment(json.loads(args.config.read_text()));model.configure('cpu');model.select(config)
    _,parents=data.read_view(args.packet);parent=next(p for p in parents if p['entry']['split']=='train');sample=data.sample(parent,0)
    rng=torch.random.get_rng_state().clone()
    train=[p for p in parents if p['entry']['split']=='train']
    amplitude=analytical.training_correction_rms(data.sample(p,slot) for p in train for slot in range(len(p['frames']))) if config['schema'] in ('rheon-water-residual-experiment-v5','rheon-water-residual-experiment-v6') else None
    samples,order,evidence=analytical.bank(sample,config['analytical_feedback'],config['training']['updates'],amplitude_pa=amplitude)
    records=[]
    with torch.no_grad():
        for index,value in enumerate(samples):
            actual=model.full_operator(value['reference'],value['cell_size_m'],value['rho'])
            relative=float(torch.linalg.vector_norm(actual-value['rhs'])/torch.linalg.vector_norm(value['rhs']))
            if not relative<1e-11:raise ValueError('analytical labels disagree with sealed finite-volume operator')
            records.append({'entry':index,'relative_operator_error':relative,'rhs_sum':float(value['rhs'].sum()),
                'pressure_gauge':float(value['reference'][0,0,0,0,0])})
    descriptor=importlib.util.spec_from_file_location('retained_legacy_analytical',args.legacy_source)
    legacy=importlib.util.module_from_spec(descriptor);descriptor.loader.exec_module(legacy)
    old_config=json.loads(args.legacy_config.read_text())
    before=legacy.bank(sample,old_config['analytical_feedback'],old_config['training']['updates'])
    after=analytical.bank(sample,old_config['analytical_feedback'],old_config['training']['updates'])
    if before[2]!=after[2] or not (before[1]==after[1]).all() or not all(torch.equal(a[k],b[k]) for a,b in zip(before[0],after[0]) for k in a):
        raise ValueError('historical bank semantics changed')
    if not torch.equal(rng,torch.random.get_rng_state()):raise ValueError('bank consumed model RNG')
    result={'schema':'rheon-multigrid-bank-qualification-v1','configuration_sha256':digest(args.config),
        'source_sha256':digest(Path(analytical.__file__)),'legacy_source_sha256':digest(args.legacy_source),
        'legacy_config_sha256':digest(args.legacy_config),'legacy_bank_tensor_order_evidence_parity':True,
        'torch_rng_preserved':True,'optimizer_updates':0,'cuda_initialized':torch.cuda.is_initialized(),
        'bank':evidence,'samples_checked':len(records),'worst_relative_operator_error':max(r['relative_operator_error'] for r in records),
        'source_train_simulations':len(train),'source_train_states':sum(len(p['frames']) for p in train),
        'checks':records,'scope':'cosine eigenvalue labels versus implemented FV operator; does not qualify a fluid trajectory'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('samples_checked','worst_relative_operator_error','legacy_bank_tensor_order_evidence_parity','torch_rng_preserved')}),flush=True)


if __name__=='__main__':main()
