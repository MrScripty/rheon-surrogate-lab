"""Replay actual training scenes with every native step and preserved force metadata."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import numpy as np
from policy import HERE, LAB, MIB, canonical, decode, fresh, load_module, read, require, sha, write


def build(prepared,cargo,output,reservation):
    paths_before=list(sys.path)
    try:
        sys.path.insert(0,str(LAB/'experiments/local-pressure-v1'))
        dependency=load_module('preview_dependency',LAB/'experiments/local-pressure-v1/dependency.py')
    finally:sys.path[:]=paths_before
    native_receipt=dependency.verify(prepared);output=fresh(output)
    paths=[LAB/'probes/water-preview-v1'/name for name in ('Cargo.toml','Cargo.lock','build.rs','src/main.rs')]+[HERE/'preview.py']
    env={k:v for k,v in os.environ.items() if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    env.update(CARGO_HOME=str(LAB/'.cargo-local'),CARGO_TARGET_DIR=str(LAB/'runs/local-pressure-native-target'),RUSTUP_TOOLCHAIN='1.92.0',RUSTC=str(cargo.parent/'rustc'),RUSTDOC=str(cargo.parent/'rustdoc'),RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='',RAYON_NUM_THREADS='1')
    common=['--manifest-path',str(LAB/'probes/water-preview-v1/Cargo.toml'),'--config',f'patch."https://github.com/MrScripty/Rheon.git".rheon.path={json.dumps(str(prepared))}','--offline','--locked']
    metadata=decode(subprocess.check_output([str(cargo),'metadata',*common,'--format-version','1'],env=env));write(output/'cargo-metadata.json',metadata)
    binding={'schema':'rheon-water-timeline-preview-build-v1','kind':'preview','prepared_source':str(prepared),'native_receipt':native_receipt,
        'metadata_sha256':sha(canonical(metadata)),'reservation_sha256':sha(read(reservation,4*MIB)),
        'source_sha256':{p.relative_to(LAB).as_posix():sha(read(p,2*MIB)) for p in paths}}
    write(output/'binding.json',binding)
    env.update(RHEON_CAPTURE_RESERVATION=str(reservation),RHEON_BUILD_BINDING=str(output/'binding.json'),RHEON_BUILD_BINDING_SHA256=sha(read(output/'binding.json')))
    for operation in ('build','test','clippy'):
        subprocess.run([str(cargo),operation,*common,'--release',*(['--','-D','warnings'] if operation=='clippy' else [])],env=env,check=True,timeout=300)
    binary=output/'rheon-water-preview';shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/rheon-water-preview',binary);binary.chmod(0o555)
    identity=decode(subprocess.check_output([str(binary),'--identity'],timeout=10));require(identity['binding']==binding,'preview_identity','exact compiled replay source')
    write(output/'build-receipt.json',{'status':'qualified_build','binary_sha256':sha(read(binary)),'identity':identity})
    return binary


def replay(binary,capture,trajectory_id,output,steps=64,stride=1):
    require(64<=steps<=600 and 1<=stride<=16,'preview_budget','bounded native continuation')
    selected=[i for i in range(steps) if (i+1)%stride==0 or i in (15,63)]
    require(len(selected)<=128,'preview_budget','at most 128 recorded states')
    paths_before=list(sys.path)
    try:
        sys.path.insert(0,str(LAB/'experiments/global-pressure-v1'))
        legacy=load_module('preview_legacy_process',LAB/'experiments/global-pressure-v1/capture.py')
    finally:sys.path[:]=paths_before
    original_raw=read(capture/'manifest.json',4*MIB);original=decode(original_raw)
    trajectory=next(t for t in original['trajectories'] if t['trajectory_id']==trajectory_id)
    require(trajectory['entry']['split']=='train','preview_population','training scenes only')
    build_receipt=decode(read(binary.parent/'build-receipt.json',2*MIB));identity=decode(subprocess.check_output([str(binary),'--identity'],timeout=10))
    require(identity==build_receipt['identity'] and sha(read(binary))==build_receipt['binary_sha256'] and identity['reservation_sha256']==original['reservation_sha256'],'preview_identity','same native reservation and replay binary')
    for path,digest in identity['binding']['source_sha256'].items():require(sha(read(LAB/path,2*MIB))==digest,'preview_source',path)
    output=fresh(output);counts=trajectory['geometry']['counts_xyz'];nz,ny,nx=counts[::-1];frames=[];parity={}
    with (output/'producer.stderr.log').open('xb') as stderr:
        with legacy.bounded_process([str(binary),'--produce'],120,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,env={**os.environ,'RAYON_NUM_THREADS':'1'}) as process:
            process.stdin.write(canonical({'trajectory_id':trajectory_id,'fixture':False,'preview':True,'preview_steps':steps,'stride':stride}));process.stdin.close()
            for index in selected:
                size=struct.unpack('<I',legacy.read_exact(process.stdout,4))[0];require(0<size<=65536,'preview_header','bounded native header')
                header=decode(legacy.read_exact(process.stdout,size));require(header['kind']=='frame' and header['index']==index,'preview_order','contiguous same-scene timesteps')
                values={};raw_fields={}
                for field in header['arrays']:
                    require(0<field['bytes']<=MIB,'preview_field','bounded native field')
                    raw=legacy.read_exact(process.stdout,field['bytes']);raw_fields[field['name']]=raw
                    array=np.frombuffer(raw,dtype=field['dtype']).reshape(field['shape_xyz'][::-1]);require(np.isfinite(array).all(),'preview_finite',field['name']);values[field['name']]=array
                if index in [f['index'] for f in trajectory['frames']]:
                    old=next(f for f in trajectory['frames'] if f['index']==index)
                    exact=all(sha(raw_fields[name])==spec['sha256'] for name,spec in old['arrays'].items())
                    require(exact,'preview_numeric_parity','all originally retained fields stay bit-exact');parity[str(index)]=exact
                u=values['corrected_u'].astype('<f8');v=values['corrected_v'].astype('<f8');w=values['corrected_w'].astype('<f8')
                velocity=np.stack(((u[:,:,:-1]+u[:,:,1:])*.5,(v[:,:-1,:]+v[:,1:,:])*.5,(w[:-1,:,:]+w[1:,:,:])*.5),axis=-1)
                frames.append({'index':index,'time_s':header['time_s'],'force_time_s':index*(1/60),'tracer':values['tracer'].reshape(-1).tolist(),
                    'pressure':values['pressure'].reshape(-1).tolist(),'velocity':velocity.reshape(-1,3).tolist()})
            size=struct.unpack('<I',legacy.read_exact(process.stdout,4))[0];require(0<size<=65536,'preview_header','bounded completion')
            completion=decode(legacy.read_exact(process.stdout,size));require(completion=={'kind':'complete','steps':steps,'snapshots':len(selected),'native_steps_qualified':steps},'preview_complete','every native step qualified')
            require(process.stdout.read(1)==b'' and process.wait(timeout=3)==0,'preview_process','drained producer')
    first=np.asarray(frames[0]['tracer']);last=np.asarray(frames[-1]['tracer']);delta=float(np.max(np.abs(last-first)))
    session={'schema':'rheon-water-training-timeline-v1','trajectory_id':trajectory_id,'counts_xyz':counts,'cell_size_m':1/trajectory['entry']['cells_per_metre'],
        'density_kg_m3':1000.0,'dynamic_viscosity_pa_s':.001,'dt_s':1/60,'force':trajectory['entry']['force'],'frames':frames,
        'provenance':{'binary_sha256':build_receipt['binary_sha256'],'capture_manifest_sha256':sha(original_raw),'retained_frame_parity':parity,'native_steps_qualified':steps,'tracer_max_abs_change':delta}}
    raw=canonical(session);require(len(raw)<=32*MIB,'preview_budget','complete bounded JSON timeline');(output/'session.json').write_bytes(raw)
    report={'status':'verified_native_replay','trajectory_id':trajectory_id,'counts_xyz':counts,'frames':len(selected),'native_steps':steps,'retained_frame_parity':parity,'tracer_max_abs_change':delta,'bytes':len(raw),'session_sha256':sha(raw)}
    write(output/'receipt.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build')
    for name in ('prepared','cargo','output','reservation'):b.add_argument('--'+name,type=Path,required=True)
    r=sub.add_parser('replay')
    for name in ('binary','capture','output'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--trajectory-id',required=True);r.add_argument('--steps',type=int,default=64);r.add_argument('--stride',type=int,default=1);a=parser.parse_args()
    if a.command=='build':result={'binary':str(build(a.prepared.absolute(),a.cargo.absolute(),a.output.absolute(),a.reservation.absolute()))}
    else:result=replay(a.binary.absolute(),a.capture.absolute(),a.trajectory_id,a.output.absolute(),a.steps,a.stride)
    print(json.dumps(result,sort_keys=True))
