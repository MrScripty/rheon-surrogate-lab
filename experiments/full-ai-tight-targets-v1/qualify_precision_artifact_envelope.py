"""Replay the actual output verifier on the trainer's exact copied normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'))
import recipe_contract


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();assembly=json.loads((args.project/'assembly.json').read_text())
    source=ROOT/assembly['static_inputs']['normalization']['path']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=assembly['static_inputs']['normalization']['sha256']:
        raise ValueError('Exact admitted normalization source changed')
    for phase in ('train','restore'):
        recipe=json.loads((args.project/(phase+'-recipe.json')).read_text())
        rule=next(row for row in recipe['outputs']if row['path']=='normalization.json')
        with tempfile.TemporaryDirectory(prefix='rheon-output-contract-',dir='/tmp')as directory:
            shutil.copyfile(source,Path(directory)/'normalization.json')
            try:recipe_contract.verify_outputs(Path(directory),[rule],recipe['resources']['max_output_bytes'])
            except Exception as error:
                print(json.dumps(dict(signal='RED',phase=phase,actual_bytes=source.stat().st_size,declared_max_bytes=rule['max_bytes'],error=str(error))))
                raise SystemExit(1)
        print(json.dumps(dict(signal='GREEN',phase=phase,actual_bytes=source.stat().st_size,declared_max_bytes=rule['max_bytes'],actual_publication_verifier=True)))
