"""Decode every original fitting field independently after final timing drains."""
import argparse,hashlib,sys
from pathlib import Path
from v6_support import ROOT,BASE,read,write,sha,fit_record
def run(output):
    if output.exists():raise ValueError('Fresh complete-bank verification required')
    project=ROOT/fit_record('v6-replay')['project'];sys.path.insert(0,str(project));import methods_view
    view=ROOT/'runs/pressure-methods-v3/derived-training-view-01';manifest=read(view/'manifest.json')
    for arm in ['v6-replay','v6-unrolled','v6-stencil','v6-separable']:
        slots={r['id']:ROOT/r['path']for r in read(ROOT/fit_record(arm)['project']/'project.json')['data']}
        if sha(slots['splits'])!=sha(view/'manifest.json'):raise ValueError('Training arms do not share the original native manifest')
    parents=methods_view.decode(tuple((view/s['path']).read_bytes()for s in manifest['shards']),manifest);rows=[]
    for i,parent in enumerate(parents):
        fields={}
        for name,array in parent['arrays'].items():
            actual=hashlib.sha256(array.tobytes()).hexdigest()
            if actual!=manifest['arrays'][f'p{i:04d}_{name}']['sha256']:raise ValueError('Decoded original field differs')
            fields[name]=actual
        rows.append(dict(trajectory_id=parent['trajectory_id'],fields=fields))
    if len(rows)!=34 or sum(len(r['fields'])for r in rows)!=136:raise ValueError('Complete34-parent136-field bank required')
    write(output,dict(status='verified',parents=34,full_fields=136,rows=rows,manifest_sha256=sha(view/'manifest.json'),source_sha256=sha(__file__)))
    print('Independently decoded all136 original fitting fields; every arm uses the same bank',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
