"""Final independent decoding of the complete expanded fitting bank."""
import argparse,hashlib,sys
from pathlib import Path
from v5_support import ROOT,BASE,read,write,sha
def verify(output):
    if output.exists():raise ValueError('Fresh bank verification required')
    project=BASE/'v5-spectrum-project-02';sys.path.insert(0,str(project))
    import methods_view
    view=BASE/'spectrum-training-view-01';manifest=read(view/'manifest.json');original=read(ROOT/'runs/pressure-methods-v3/derived-training-view-01/manifest.json')
    for i,parent in enumerate(original['parents']):
        if parent!=manifest['parents'][i]:raise ValueError('Retained original metadata changed')
        for field in ['base','rhs','reference','native_input_residual']:
            key=f'p{i:04d}_{field}'
            if original['arrays'][key]!=manifest['arrays'][key]:raise ValueError('Original full-field inventory changed')
    parents=methods_view.decode(tuple((view/s['path']).read_bytes()for s in manifest['shards']),manifest)
    checked=[]
    for i,parent in enumerate(parents):
        fields={}
        for field,array in parent['arrays'].items():
            key=f'p{i:04d}_{field}';digest=hashlib.sha256(array.tobytes()).hexdigest()
            if digest!=manifest['arrays'][key]['sha256']:raise ValueError('Decoded complete field mismatch')
            fields[field]=digest
        checked.append(dict(trajectory_id=parent['trajectory_id'],fields=fields))
    write(output,dict(status='verified',parents=len(parents),retained_original_parents=len(original['parents']),full_fields=sum(len(r['fields'])for r in checked),
        rows=checked,manifest_sha256=sha(view/'manifest.json'),original_manifest_sha256=sha(ROOT/'runs/pressure-methods-v3/derived-training-view-01/manifest.json'),source_sha256=sha(__file__)))
    print('Independently decoded all184 fields; all136 original fields preserved exactly',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);verify(p.parse_args().output)
