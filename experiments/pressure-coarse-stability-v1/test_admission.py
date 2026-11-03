"""Isolated negatives reach the intended screen guard before any owner mutation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from run_stability import validate_admission


def write(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','assembly','capture-verification','probe-qualification','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    selected=parser.parse_args();selected.root=selected.root.resolve();selected.output=selected.output.resolve();selected.output.mkdir(parents=True)
    valid=SimpleNamespace(**vars(selected))
    assert validate_admission(valid)==1500
    assembly=json.loads(valid.assembly.read_text());project=json.loads((valid.root/assembly['project']['path']).read_text())
    recipe_row=next(r for r in project['recipes']if r['id']=='train')
    recipe=json.loads((valid.root/recipe_row['path']).read_text())
    results=[]
    for label in ('ordinary-entry','cap-zero','cap-5000','cap-missing','cap-duplicate','qualification-binding'):
        directory=selected.output/label;directory.mkdir()
        args=SimpleNamespace(**vars(valid))
        if label=='qualification-binding':
            value=json.loads(valid.probe_qualification.read_text());value['clipping_sha256']='0'*64
            args.probe_qualification=directory/'qualification.json';write(args.probe_qualification,value)
            expected='Bound integrated repair/capture qualification required before admission'
        else:
            p=copy.deepcopy(project);r=copy.deepcopy(recipe);a=copy.deepcopy(assembly)
            if label=='ordinary-entry':
                r['entrypoint']=str(Path(r['entrypoint']).with_name('evidence.py'))
                expected='Qualified clipping stability entry required'
            else:
                index=r['arguments'].index({'literal':'--gradient-limit'})
                if label=='cap-zero':r['arguments'][index+1]={'literal':'0'}
                if label=='cap-5000':r['arguments'][index+1]={'literal':'5000'}
                if label=='cap-missing':del r['arguments'][index:index+2]
                if label=='cap-duplicate':r['arguments'] += [{'literal':'--gradient-limit'},{'literal':'1500'}]
                expected='Exact1500-update screen required'if label in('cap-zero','cap-5000')else'Unique explicit screen argument required: --gradient-limit'
            recipe_path=directory/'train-recipe.json';write(recipe_path,r)
            row=next(row for row in p['recipes']if row['id']=='train');row.update(path=recipe_path.relative_to(valid.root).as_posix(),sha256=digest(recipe_path))
            path=directory/'project.json';write(path,p)
            a['project'].update(path=path.relative_to(valid.root).as_posix(),sha256=digest(path))
            args.assembly=directory/'assembly.json';write(args.assembly,a)
        try:validate_admission(args)
        except ValueError as error:
            assert str(error)==expected
            results.append(dict(case=label,expected=expected,observed=str(error),passed=True))
        else:raise AssertionError('Unsafe admission accepted: '+label)
    result=dict(status='passed',valid_capture_accepted=True,negative_cases=results,owner_mutations=0,
        source_sha256=digest(Path(__file__)),assembly_sha256=digest(valid.assembly),
        validation_source_sha256=digest(Path(__file__).with_name('run_stability.py')))
    write(selected.output/'receipt.json',result);print(json.dumps(result))


if __name__=='__main__':main()
