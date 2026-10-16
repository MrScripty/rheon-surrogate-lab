"""Export qualified CPU-teacher pressure timelines to existing Tuldok fields."""
import argparse
import io
import hashlib
import json
from pathlib import Path
import sys
import uuid
import zipfile

import numpy as np
from diagnose import LAB,digest,read,write


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=read(args.report);root=LAB/report['output_root'];args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(LAB/'runs/live-integration/Tuldok'))
    from adapters.rheon.generate import npy
    import scientific_fields
    rows=[]
    for index,record in enumerate(report['parents']):
        assert record['status']=='qualified_cpu_teacher_parent' and record['network_calls']==0
        directory=root/f'parent-{index:02d}';assert digest(directory/'fields.npz')==record['fields_sha256']
        scene=record['scene'];h=1/scene['cells_per_metre'];members={}
        identity=f'targeted-cpu--parent-{index:02d}'
        manifest={'schema_version':1,'format':'scientific_field_sequence_v1',
            'grid':{'shape_xyz':scene['counts'],'origin_xyz':[0,0,0],'spacing_xyz':[h]*3,
                    'orientation':[[1,0,0],[0,1,0],[0,0,1]],'units':'m'},'time_units':'s',
            'fields':[{'id':name,'label':label,'dtype':'<f8','centering':'cell','units':'Pa',
                       'stage':'Native water dynamics with independent CPU spectral pressure teacher; no neural inference'}
                      for name,label in [('pressure','CPU teacher pressure'),('base','Previous pressure')]],
            'frames':[],'family':{'namespace':'sealed-water-forcing-v1','id':identity},'sources':[],
            'provenance':{'producer':'Rheon native water dynamics / CPU spectral teacher','source_id':record['fields_sha256']}}
        with np.load(directory/'fields.npz',allow_pickle=False) as fields:
            for sample in record['samples']:
                step=sample['step'];row={'time':str(step/60),'arrays':[]}
                for key in ('pressure','base'):
                    value=fields[f'{step:03d}-{key}'];assert list(value.shape)==scene['counts'][::-1]
                    path=f'arrays/{step:03d}-{key}.npy';raw=npy(value.astype('<f8').tobytes(),'<f8',value.shape);members[path]=raw
                    row['arrays'].append({'field':key,'path':path,'bytes':len(raw),'sha256':sha(raw)})
                manifest['frames'].append(row)
        source=encoded({'schema':'rheon-cpu-teacher-parent-provenance-v1','scene':scene,
            'material':{'density_kg_m3':1000.,'dynamic_viscosity_pa_s':.001,'dt_s':1/60,'boundary':'sealed_impermeable_free_slip'},
            'receipt_sha256':digest(directory/'receipt.json'),'native_frames_sha256':digest(directory/'frames.jsonl'),
            'field_sha256':record['fields_sha256'],'label_source':record['label_source'],'network_calls':0,
            'scope':'future training parent; pressure/base/residual arrays locally retained; no current-fit input or velocity/tracer export'})
        members['scene-provenance.json']=source
        manifest['sources']=[{'path':'scene-provenance.json','bytes':len(source),'sha256':sha(source),'description':'Actual force, cubic geometry, material, native physics and teacher lineage'}]
        members['manifest.json']=encoded(manifest);stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
            for name,value in sorted(members.items()):z.writestr(name,value)
        raw=stream.getvalue();scientific_fields.parse_bundle(raw)
        path=args.output/(identity+'.zip');path.write_bytes(raw)
        rows.append({'trajectory_id':identity,'split':'train','counts_xyz':scene['counts'],
                     'path':str(path.relative_to(LAB)),'sha256':sha(raw),'bytes':len(raw)})
    project={'schema_version':1,'id':uuid.uuid4().hex,'name':'Future water parents · CPU spectral teacher',
        'activity_path':'runs/live-project/activity.json','data':[{'id':r['trajectory_id'],
        'label':r['trajectory_id']+' · CPU teacher · 49 evolving frames','path':r['path'],'sha256':r['sha256']} for r in rows],
        'recipes':[],'evidence':[],'requirements':[]}
    path=args.output/'project.json';write(path,project)
    write(args.output/'exports.json',{'population':'train_validation','rows':rows,'projects':[str(path.relative_to(LAB))]})
    print({'parents':len(rows),'field_frames':sum(len(r['samples']) for r in report['parents'])})


if __name__=='__main__':main()
