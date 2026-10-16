"""Qualify complete paired banks and exact historical primary stream parity."""
import argparse
from importlib.util import module_from_spec,spec_from_file_location
import time
from pathlib import Path

from diagnose import LAB,digest,read,write
import sys
sys.path.insert(0,str(LAB/'experiments/water-conditioning-v1/training'))
import analytical
import data
import model
import policy
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();started=time.perf_counter();sequence=read(args.sequence)
    model.configure('cpu');_,parents=data.read_view(LAB/sequence['training_packet'])
    training=[p for p in parents if p['entry']['split']=='train']
    sample=data.sample(training[0],0)
    amplitude=analytical.training_correction_rms(data.sample(p,i) for p in training for i in range(len(p['frames'])))
    path=LAB/'runs/water-coverage-feedback-v1/feedback-only/project/analytical.py'
    descriptor=spec_from_file_location('captured_v6_bank',path);legacy=module_from_spec(descriptor);descriptor.loader.exec_module(legacy)
    rows=[]
    for trial in sequence['trials']:
        config=policy.experiment(read(LAB/trial['config']));model.select(config);rng=torch.random.get_rng_state().clone()
        primary=analytical.bank(sample,config['analytical_feedback'],config['training']['updates'],amplitude_pa=amplitude)
        before=legacy.bank(sample,config['analytical_feedback'],config['training']['updates'],amplitude_pa=amplitude)
        assert primary[2]==before[2] and (primary[1]==before[1]).all()
        assert all(torch.equal(a[k],b[k]) for a,b in zip(primary[0],before[0]) for k in a)
        supplement=analytical.supplemental_bank(sample,config['supplemental_feedback'],config['training']['updates'],primary,config['analytical_feedback'],amplitude)
        worst=0.;checked=0
        for values in (primary[0],supplement[0]):
            for value in values:
                rhs=value['rhs'];error=model.full_operator(value['reference'],value['cell_size_m'],value['rho'])-rhs
                relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(rhs));assert relative<1e-11
                worst=max(worst,relative);checked+=1
        assert torch.equal(rng,torch.random.get_rng_state())
        def brief(bank):return {k:v for k,v in bank[2].items() if k!='grids'}
        rows.append({'trial':trial['name'],'config_sha256':digest(LAB/trial['config']),
            'primary_tensor_order_evidence_parity':True,'torch_rng_preserved':True,
            'primary':brief(primary),'supplemental':brief(supplement),'labels_checked_including_reuse':checked,
            'worst_relative_operator_error':worst})
    write(args.output,{'trials':rows,'train_parents':len(training),'train_states':sum(len(p['frames']) for p in training),
        'amplitude_rms_pa':amplitude,'legacy_source_sha256':digest(path),'source_sha256':digest(Path(analytical.__file__)),
        'optimizer_updates':0,'cuda_initialized':torch.cuda.is_initialized(),'elapsed_seconds':time.perf_counter()-started})
    print({r['trial']:{'labels':r['labels_checked_including_reuse'],'worst':r['worst_relative_operator_error'],
                       'combined_bytes':r['supplemental']['combined_resident_tensor_bytes']} for r in rows})


if __name__=='__main__':main()
