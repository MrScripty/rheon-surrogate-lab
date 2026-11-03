"""Paired sampling repeats of a frozen selected method, retaining initialization."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Replica builder seam changed: '+before[:120])
    return text.replace(before,after)

def prepare(arm,seed):
    selection_path=ROOT/'docs/plans/pressure-methods-v2/selection.json'
    selection=read(selection_path)
    if arm not in [r['arm'] for r in selection['selected']] or seed not in [2026101825,2026101826]:
        raise ValueError('Only selected methods and declared paired seeds may repeat')
    base=ROOT/'runs/pressure-methods-v2';identifier=arm+'-s'+str(seed)[-2:]
    original=base/(arm+'-builder-02.py')
    if sha(original)!=next(r['builder_sha256'] for r in selection['selected'] if r['arm']==arm):
        raise ValueError('Selected builder identity changed')
    text=original.read_text()
    text=replace(text,'training_seed=2026101820, sampling_seed=2026101804,',f'training_seed={seed}, sampling_seed={seed},')
    text=replace(text,"config_path = output.parent/(arm+'-experiment-02.json')",f"config_path = output.parent/{(identifier+'-experiment.json')!r}")
    text=replace(text,"captures = output.parent/(arm+'-source-overrides-02')",f"captures = output.parent/{(identifier+'-source-overrides')!r}")
    text=replace(text,"project['name'] = 'Pressure methods · '+arm+' · C53'",
        f"project['name'] = 'Pressure methods · '+arm+' · C53 · seed {seed}'")
    capture=base/(identifier+'-builder.py')
    if capture.exists():raise ValueError('Fresh repeat required; never replay an admitted trial')
    capture.write_text(text)
    spec=importlib.util.spec_from_file_location('v2_selected_repeat_builder',capture)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    output=base/(identifier+'-project-01')
    builder.build(arm,output)
    initial=read(output/'project.json')
    reference=read(base/(arm+'-project-02/project.json'))
    def identities(project):
        return {r['id']:r['sha256'] for r in project['data'] if r['id'] in ['dataset','dataset_extra','splits','initial_weights']}
    if identities(initial)!=identities(reference):raise ValueError('Repeat changed data or checkpoint initialization')
    write(output/'repeat.json',dict(status='prepared_not_admitted',identifier=identifier,parent_arm=arm,
        selection_sha256=sha(selection_path),seed=seed,original_builder_sha256=sha(original),
        captured_builder_sha256=sha(capture),source_sha256=sha(__file__),
        identical_data_and_initialization=identities(initial),sampling_and_training_seed_changed=True))
    print('Prepared selected sampling repeat',identifier,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--seed',type=int,choices=[2026101825,2026101826],required=True)
    args=parser.parse_args();prepare(args.arm,args.seed)
