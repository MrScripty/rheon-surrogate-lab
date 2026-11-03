"""Initialize only the new tail; preserve every verified parent tensor exactly."""
import argparse
from pathlib import Path
import sys
import numpy as np
from study_support import ROOT,read,write,sha


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('project','initial-output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();project=args.project.resolve();output=args.initial_output.resolve()
    if output.exists():raise ValueError('Fresh seeded initialization required')
    context=read(project/'initial-context.json');parent=read(ROOT/context['parent_entry'])
    for name,digest in parent['identities'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Parent CPU component identity changed')
    artifacts=ROOT/parent['restore'];parent_receipt=read(artifacts/'assembly-receipt.json')
    if parent_receipt['mode']!='local_weight_assembly'or parent_receipt['optimizer_updates']!=0:
        raise ValueError('Verified parent assembly provenance required')
    sys.path.insert(0,str(project));import model,torch,head_utils
    calibration=read(project/'calibration.json');config=read(project/'config.json')
    model.configure('cpu');torch.set_num_threads(1);torch.manual_seed(calibration['seed']);model.select(config)
    network=model.PressureCorrectionNet().eval()
    with np.load(artifacts/'weights.npz',allow_pickle=False)as source:
        values={name:torch.from_numpy(source[name].copy())for name in source.files}
    loaded=network.load_state_dict(values,strict=False)
    expected={'tail_logits','tail_shape_head.0.weight','tail_shape_head.0.bias','tail_shape_head.2.weight','tail_shape_head.2.bias'}
    if set(loaded.missing_keys)!=expected or loaded.unexpected_keys:
        raise ValueError('Only new tail parameters may lack parent tensors')
    bank=ROOT/context['teacher_bank']
    with np.load(bank/'teacher.npz',allow_pickle=False)as source:
        chosen=source['split']==0
        valid=source['valid_mode'][chosen].astype('f8')
        fraction=np.exp(source['log_gain_fraction'][chosen].astype('f8'))
    logits=np.log(fraction/(1-fraction))
    mean=(logits*valid).sum(0)/valid.sum(0)
    with torch.no_grad():network.tail_logits.copy_(torch.from_numpy(mean.astype('f4')))
    inventory=head_utils.inventory(network)
    parent_inventory={name:row for name,row in inventory.items()if name not in expected}
    if parent_inventory!=parent_receipt['inventory']:
        raise ValueError('Parent parameter bytes or inventory differ after initialization')
    if sum(p.numel()for p in network.parameters())!=128938 or torch.cuda.is_initialized():
        raise ValueError('Expected128938CPUfloat32parameters')
    output.mkdir();head_utils.save_weights(network,output/'weights.npz')
    write(output/'initial-inventory.json',dict(inventory=inventory,architecture=model.ARCHITECTURE,
        weights_sha256=sha(output/'weights.npz'),normalization_sha256=sha(artifacts/'normalization.json'),
        parent_entry=context['parent_entry'],parent_entry_sha256=sha(ROOT/context['parent_entry']),
        parent_assembly_receipt_sha256=sha(artifacts/'assembly-receipt.json'),parent_weights_sha256=sha(artifacts/'weights.npz'),
        parent_tensors_bitwise_unchanged=True,tail_initialization='Train-only per-mode mean logits plus seeded hidden layer and zero final shape-head layer',
        teacher_manifest_sha256=sha(bank/'manifest.json'),training_geometries=int(chosen.sum()),
        source_sha256=sha(__file__),optimizer_updates=0,managed_training_run_claimed=False))
    print('Initialized33982newtailparameters;94956parentparametersbitwiseunchanged',flush=True)
