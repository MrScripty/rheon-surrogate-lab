"""Map exact C107 tensors to a registered architecture and freeze its inventory."""
import argparse
import sys
from study_support import ROOT,read,write,sha

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True);args=parser.parse_args()
    from pathlib import Path
    project=Path(args.project).resolve();sys.path.insert(0,str(project))
    import torch,model,student_original,data
    config=read(project/'config.json');model.configure('cpu');model.select(config)
    entry=read(ROOT/'runs/pressure-methods-v4/entries/v4-student50-control.json')
    origin=ROOT/entry['restore'];source=origin/'weights.npz'
    if sha(source)!=entry['weights_sha256']:raise ValueError('Exact C107 weights required')
    original_inventory=read(origin/'history.json')['checkpoints']['checkpoint-2400.npz']['inventory']
    torch.manual_seed(config['training']['seed'])
    network=student_original.PressureCorrectionNet()
    data.helpers().load_weights(network,source,original_inventory)
    network=model.transform(network)
    inventory=data.helpers().state_inventory(network)
    import numpy as np
    target=project/'architecture-initial.npz'
    np.savez(target,**{name:value.detach().numpy()for name,value in network.state_dict().items()})
    objective=read(project/'fine-tuning.json')
    objective.update(initial_weights_sha256=sha(target),initial_inventory=inventory)
    write(project/'fine-tuning.json',objective)
    document=read(project/'project.json')
    slot=next(r for r in document['data']if r['id']=='initial_weights')
    slot.update(path=str(target.relative_to(ROOT)),sha256=sha(target))
    write(project/'project.json',document)
    write(project/'architecture-initialization.json',dict(status='mapped_from_verified_C107',
        original_weights_sha256=sha(source),mapped_weights_sha256=sha(target),inventory=inventory,
        parameters=sum(p.numel()for p in network.parameters()),source_sha256=sha(__file__),
        initialization_function_equivalence_claimed=False,optimizer_updates=0))
    print('Mapped C107 architecture',project.name,'parameters',sum(p.numel()for p in network.parameters()),flush=True)
