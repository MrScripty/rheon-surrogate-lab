"""Lossless variable-parent water packets with explicit reserved test separation."""
import math
import io
from pathlib import Path
import numpy as np
import torch
from policy import HERE, MIB, canonical, decode, fresh, read, require, sha, write
import scenes
import model

FIELDS=('base','reference','rhs','native_input_residual')


def helpers():return model.frozen.load_original('train',model.frozen.ORIGINAL_TRAIN_SHA256)


def inventory(arrays):
    return {name:{'shape':list(v.shape),'dtype':v.dtype.str,'bytes':v.nbytes,'sha256':sha(v.tobytes())} for name,v in arrays.items()}


def load(raw,manifest):
    require(type(manifest) is dict and set(manifest)=={'schema','reservation','reservation_sha256','dataset_sha256','native_build_receipt','parents','arrays','population'},'packet','closed water packet')
    require(manifest['schema']=='rheon-sealed-water-packet-v1' and sha(raw)==manifest['dataset_sha256'],'packet','selected lossless payload')
    reservation=scenes.validate_reservation(manifest['reservation']);config=reservation['configuration']
    require(sha(canonical(reservation))==manifest['reservation_sha256'] and manifest['population'] in ('train_validation','reserved_test'),'population','explicit independent population')
    rows=[r for r in reservation['scenes'] if (r['split']=='reserved_test')==(manifest['population']=='reserved_test')]
    require(len(rows)==len(manifest['parents']),'parents','whole reserved scene coverage')
    declared={}
    for index,(parent,row) in enumerate(zip(manifest['parents'],rows)):
        require(parent['scene']==row and parent['entry']=={'trajectory_id':row['trajectory_id'],'split':row['split'],'dt_s':config['dt_s']}
                and parent['counts_xyz']==row['counts_xyz'] and parent['cell_size_m']==1/row['cells_per_metre']
                and parent['density_kg_m3']==config['material']['density_kg_m3'],'physical_lineage','exact scene, material and cubic spacing')
        require([f['index'] for f in parent['frames']]==config['snapshot_steps'],'frame_population','reserved snapshots')
        for field in FIELDS:
            key=f'p{index:03d}_{field}';spec=manifest['arrays'][key]
            require(spec['dtype']=='<f8' and spec['shape']==[len(config['snapshot_steps']),*row['counts_xyz'][::-1]]
                    and spec['bytes']==len(config['snapshot_steps'])*math.prod(row['counts_xyz'])*8,'shape',key);declared[key]=spec
    require(set(declared)==set(manifest['arrays']),'numeric_inventory','closed packet arrays')
    arrays=helpers().read_npz(raw,declared,32*MIB);parents=[]
    for index,parent in enumerate(manifest['parents']):
        values={field:arrays[f'p{index:03d}_{field}'] for field in FIELDS}
        require(bool((values['base'][:,0,0,0]==0).all() and (values['reference'][:,0,0,0]==0).all()),'gauge','cell zero')
        for slot,frame in enumerate(parent['frames']):
            require(set(frame['sha256'])==set(FIELDS),'frame_inventory','all training fields')
            for field in FIELDS:require(sha(values[field][slot].tobytes())==frame['sha256'][field],'raw_lineage',field)
        parents.append({**parent,'arrays':values})
    return reservation,parents


def sample(parent,index,labels=True,device='cpu'):
    fields=FIELDS if labels else ('base','native_input_residual')
    result={name:torch.from_numpy(parent['arrays'][name][index:index+1].copy()).unsqueeze(1) for name in fields}
    result.update(cell_size_m=torch.tensor([parent['cell_size_m']],dtype=torch.float64),
        rho=torch.tensor([parent['density_kg_m3']],dtype=torch.float64),dt=torch.tensor([parent['entry']['dt_s']],dtype=torch.float64))
    return {name:value.to(device) for name,value in result.items()}


def fit_normalization(parents):
    trains=[p for p in parents if p['entry']['split']=='train'];require(trains,'training_population','train only')
    cells=sum(p['arrays']['base'].size for p in trains);scales={}
    for name,field,units in [('residual','native_input_residual','m3/s2'),('pressure','base','Pa'),('rhs','rhs','m3/s2')]:
        rms=math.sqrt(math.fsum(float(np.square(p['arrays'][field]).sum(dtype=np.float64)) for p in trains)/cells)
        require(math.isfinite(rms) and rms>0,'normalization','positive train-only RMS')
        scales[name]={'value':rms,'units':units,'method':'train_cell_rms','source':field}
    for name,key,units in [('rho','density_kg_m3','kg/m3'),('dt','dt_s','s')]:
        values=[p[key] if key in p else p['entry'][key] for p in trains for _ in p['frames']];constant=all(v==values[0] for v in values)
        scales[name]={'value':1.0 if constant else math.sqrt(math.fsum(v*v for v in values)/len(values)),
            'units':units,'method':'constant_unit_scale' if constant else 'train_frame_rms','source':name}
    return {'schema':'rheon-cubic-pressure-normalization-v2','model_profile':model.ARCHITECTURE['schema'],'channels':model.CHANNELS,
        'coordinates':'actual cell centres in metres; unit scale 1 m','scales':scales,
        'training_provenance':{'trajectory_ids':[p['entry']['trajectory_id'] for p in trains],'frames':sum(len(p['frames']) for p in trains),
            'cells':cells,'grids':[p['counts_xyz'] for p in trains],
            'parent_numeric_sha256':[{f:sha(p['arrays'][f].tobytes()) for f in FIELDS} for p in trains]}}


def pack(capture_root,output):
    import capture
    report=capture.verify(capture_root);require(report['status']=='verified_development','capture','full qualified water corpus')
    raw_manifest=read(capture_root/'manifest.json',4*MIB);native=decode(raw_manifest);output=fresh(output);results={}
    for population in ('train_validation','reserved_test'):
        selected=[p for p in native['trajectories'] if (p['entry']['split']=='reserved_test')==(population=='reserved_test')]
        arrays={};parents=[]
        for index,parent in enumerate(selected):
            fields={field:np.empty((len(parent['frames']),*parent['geometry']['counts_xyz'][::-1]),dtype='<f8') for field in FIELDS}
            frames=[]
            for slot,frame in enumerate(parent['frames']):
                for field in FIELDS:
                    descriptor=frame['arrays']['pressure' if field=='reference' else field]
                    fields[field][slot]=np.frombuffer(read(capture_root/descriptor['path'],MIB),dtype='<f8').reshape(fields[field][slot].shape)
                frames.append({'index':frame['index'],'sha256':{f:sha(v[slot].tobytes()) for f,v in fields.items()}})
            for field,value in fields.items():arrays[f'p{index:03d}_{field}']=value
            geom=parent['geometry'];entry=parent['entry']
            parents.append({'scene':entry,'entry':{'trajectory_id':entry['trajectory_id'],'split':entry['split'],'dt_s':native['reservation']['configuration']['dt_s']},
                'counts_xyz':geom['counts_xyz'],'cell_size_m':1/entry['cells_per_metre'],'density_kg_m3':geom['density_kg_m3'],
                'capture_root':str(capture_root),'capture_manifest_sha256':sha(raw_manifest),'frames':frames})
        directory=fresh(output/population)
        with (directory/'dataset.npz').open('xb') as stream:np.savez(stream,**arrays)
        raw=read(directory/'dataset.npz',32*MIB)
        manifest={'schema':'rheon-sealed-water-packet-v1','population':population,'reservation':native['reservation'],'reservation_sha256':native['reservation_sha256'],
            'native_build_receipt':native['native_build_receipt'],'dataset_sha256':sha(raw),'parents':parents,'arrays':inventory(arrays)}
        write(directory/'splits.json',manifest);load(raw,manifest)
        receipt={'status':'verified_lossless_water','reader_source_sha256':sha(read(HERE/'data.py')),'dataset_sha256':sha(raw),
            'manifest_sha256':sha(read(directory/'splits.json')),'parents':len(parents),'snapshots':sum(len(p['frames']) for p in parents),'population':population}
        write(directory/'dataset-receipt.json',receipt);results[population]=receipt
    return results
