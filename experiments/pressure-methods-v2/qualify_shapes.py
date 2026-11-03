"""Reuse physical objective checks, then verify replay and optimizer ownership."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def qualify(project,output):
    original=ROOT/'experiments/pressure-methods-v1/qualify_training.py'
    text=original.read_text()
    before="    if max(exposures.values())-min(exposures.values()) > 1:\n        raise ValueError('Spatial-tier imbalance')"
    after="    if sum(exposures.values()) != config['training']['updates']:\n        raise ValueError('Exposure count differs from declared updates')"
    if text.count(before)!=1:raise ValueError('Qualification seam changed')
    text=text.replace(before,after).replace('ROOT = Path(__file__).resolve().parents[2]', 'ROOT = Path('+repr(str(ROOT))+')')
    text=text.replace('HERE = Path(__file__).resolve().parent',"HERE = ROOT/'experiments/pressure-methods-v1'")
    capture=output.with_name(output.stem+'-qualifier.py')
    if capture.exists():raise ValueError('Fresh qualification required')
    capture.write_text(text)
    spec=importlib.util.spec_from_file_location('v2_physical_qualifier',capture)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    base.qualify(project,output)
    result=read(output)
    import torch,model,data,fine_tuning,mixed_order
    config=read(project/'config.json')
    import train
    train.experiment(config)
    result['entrypoint_configuration_validated']=True
    from common import source_inputs
    paths=source_inputs(project)
    _,parents=data.load(tuple(paths[k].read_bytes() for k in ['dataset','dataset_extra']),read(paths['splits']))
    actual=mixed_order.order(parents,config)
    small=[p for p in parents if max(p['counts_xyz'])<=16]
    control_config=dict(config,training=dict(config['training'],updates=1600))
    control=mixed_order.order(small,control_config)
    retained=actual[[max(parents[int(row[0,0])]['counts_xyz'])<=16 for row in actual]]
    n=min(len(retained),1600)
    if not bool((retained[:n]==control[:n]).all()):raise ValueError('Small-control sample subsequence changed')
    result['small_replay']=dict(matching_prefix_updates=n,exact_sample_subsequence=True)
    if (project/'head_optimizer.py').exists():
        import head_optimizer
        network=model.PressureCorrectionNet()
        initial=ROOT/next(row['path'] for row in read(project/'project.json')['data'] if row['id']=='initial_weights')
        fine_tuning.initialize(network,initial,data.helpers())
        inventory=data.helpers().state_inventory(network)
        groups=head_optimizer.groups(network,config['training']['learning_rate'])
        ids=[id(p) for group in groups for p in group['params']]
        if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in network.parameters()}:
            raise ValueError('Optimizer ownership incomplete')
        sample=data.sample(parents[-1],0)
        normalization=model.fit_normalization(parents,data.fit_normalization)
        loss=model.training_loss(network,sample,normalization).mean();loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
            raise ValueError('Incomplete finite coarse/fine gradients')
        if data.helpers().state_inventory(network)!=inventory:raise ValueError('Qualification updated parameters')
        result['head_groups']=dict(parameter_counts=[sum(p.numel() for p in g['params']) for g in groups],
            lr_multipliers=[g['lr_multiplier'] for g in groups],optimizer_steps=0)
    result.update(source_sha256=sha(__file__),captured_qualifier_sha256=sha(capture))
    write(output,result)
    print('Qualified small replay and actual objective/optimizer',project.name,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();qualify(args.project.resolve(),args.output.resolve())
