"""Create the predeclared trial views from immutable qualified capture packets."""
import argparse
from pathlib import Path
import json
from policy import LAB, canonical, decode, read, require, sha, write
import views


def prepare(root, include_large):
    baseline=LAB/'runs/sealed-water-v1-3/packets/train_validation'
    destination=root/'views';destination.mkdir(exist_ok=True)
    val=[('baseline','validation',i,[0,1]) for i in range(64)]
    sources={'baseline':baseline}
    if not (destination/'validation').exists():views.create(sources,val,'validation',destination/'validation')
    for name,selected in [('fit-one',[('baseline','train',0,[0])]),('long-256',[('baseline','train',i,[0,1]) for i in range(256)]),
        ('scale-256',[('baseline','train',i,[i%2]) for i in range(256)])]:
        if not (destination/name).exists():views.create(sources,selected+val,'train_validation',destination/name)
    if include_large:
        for n in range(1,4):sources['cohort-'+str(n)]=root/f'cohort-{n}/packets/train_validation'
        training=[(source,'train',i,[i%2]) for source in sources for i in range(256)]
        if not (destination/'scale-1024').exists():views.create(sources,training+val,'train_validation',destination/'scale-1024')
        if not (destination/'reserved-test').exists():views.create({'fresh-test':root/'cohort-3/packets/reserved_test'},[('fresh-test','reserved_test',i,[0,1]) for i in range(64)],'reserved_test',destination/'reserved-test')
    return {'views':[str(p) for p in sorted(destination.iterdir())]}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--include-large',action='store_true');a=parser.parse_args()
    print(json.dumps(prepare(a.root.absolute(),a.include_large)))
