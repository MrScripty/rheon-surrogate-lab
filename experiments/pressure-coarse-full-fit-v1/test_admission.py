"""Full-fit admission rejects wrong entries, bounds and qualification witnesses."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from run_fit import validate


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','assembly','capture-verification','entry-qualification','control-comparison','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.root=args.root.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True)
    validate(args)
    assembly=json.loads(args.assembly.read_text());project=json.loads((args.root/assembly['project']['path']).read_text())
    row=next(r for r in project['recipes']if r['id']=='train');recipe=json.loads((args.root/row['path']).read_text())
    results=[]
    for label in ('capped-entry','missing-reference','duplicate-reference','capped-flag','updates1500','attempts2','entry-proof','control-proof'):
        directory=args.output/label;directory.mkdir();selected=SimpleNamespace(**vars(args))
        if label in('entry-proof','control-proof'):
            target='entry_qualification'if label=='entry-proof'else'control_comparison'
            value=json.loads(getattr(args,target).read_text())
            value['source_sha256'if label=='entry-proof'else'campaign_package_weights_bound']='0'*64 if label=='entry-proof'else False
            path=directory/'qualification.json';write(path,value);setattr(selected,target,path)
            expected='Bound full-fit/capture/control qualification required before admission'
        else:
            a=copy.deepcopy(assembly)
            if label in('updates1500','attempts2'):
                if label=='updates1500':a['updates']=1500
                else:a['limits']['attempt_slots']=2
                expected='One full5000-fit attempt required'
            else:
                p=copy.deepcopy(project);r=copy.deepcopy(recipe)
                if label=='capped-entry':
                    r['entrypoint']=str(Path(r['entrypoint']).with_name('stability_screen.py'));expected='Qualified full-fit entry required'
                elif label=='capped-flag':
                    r['arguments'] += [{'literal':'--gradient-limit'},{'literal':'1500'}];expected='Capped diagnostic cannot be a full fit'
                else:
                    i=r['arguments'].index({'literal':'--fit-reference'})
                    if label=='missing-reference':del r['arguments'][i:i+2]
                    else:r['arguments']+=r['arguments'][i:i+2]
                    expected='Unique captured full-fit reference required'
                path=directory/'train-recipe.json';write(path,r)
                ref=next(row for row in p['recipes']if row['id']=='train');ref.update(path=path.relative_to(args.root).as_posix(),sha256=digest(path))
                path=directory/'project.json';write(path,p);a['project'].update(path=path.relative_to(args.root).as_posix(),sha256=digest(path))
            selected.assembly=directory/'assembly.json';write(selected.assembly,a)
        try:validate(selected)
        except ValueError as error:
            assert str(error)==expected
            results.append(dict(case=label,expected=expected,observed=str(error),passed=True))
        else:raise AssertionError('Wrong full-fit admission accepted')
    result=dict(status='passed',valid_capture_accepted=True,negative_cases=results,owner_mutations=0,
        assembly_sha256=digest(args.assembly),source_sha256=digest(Path(__file__)),validator_sha256=digest(Path(__file__).with_name('run_fit.py')))
    write(args.output/'receipt.json',result);print(json.dumps(dict(status='passed',negative_cases=len(results),owner_mutations=0)))


if __name__=='__main__':main()
