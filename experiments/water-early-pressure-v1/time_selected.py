"""Freeze consumed timing inputs and reuse the established serial episode owner."""
import argparse
import importlib.util
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('current_serial_owner',LAB/'experiments/water-pressure-bound-regression-v1/timing.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
read,write,digest=owner.read,owner.write,owner.digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--reports',type=Path,required=True)
    args=p.parse_args();config=read(args.config);root=LAB/config['root']
    selection=read(args.reports/'regression-summary.json')
    challenger=next(m for m in config['models'] if m['name']==selection['frozen_challenger'])
    late=read(LAB/'experiments/water-pressure-bound-regression-v1/configs/regression.json')['models'][0]
    models=[late,challenger]
    bindings={'binary_sha256':digest(LAB/config['binary']),'models':{m['name']:{
        'weights_sha256':digest(LAB/m['artifacts']/'weights.npz'),
        'normalization_sha256':digest(LAB/m['artifacts']/'normalization.json'),
        'source_witness_sha256':digest(LAB/m['project']/'source-witness.json'),
        'stopping_profile_sha256':digest(LAB/m['stopping_profile'])} for m in models}}
    indices=[0,16,13]
    directory=root/'timing-configs';directory.mkdir(exist_ok=False)
    for i in indices:write(directory/f'timing-{i:02d}.json',{**config,'models':models,'serial_timing_scene':i})
    write(directory/'protocol.json',{'root':config['root'],'timing_scene_indices':indices})
    write(directory/'bindings.json',bindings)
    sys.argv=[str(Path(owner.__file__)),'--protocol',str(directory/'protocol.json'),
              '--bindings',str(directory/'bindings.json'),'--configs',str(directory),'--reports',str(args.reports)]
    owner.main()


if __name__=='__main__':main()
