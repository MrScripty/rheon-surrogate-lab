"""Bounded native trajectories with independent retained-snapshot physics checks."""
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
from policy import HERE, LAB, MIB, canonical, decode, fresh, load_module, read, require, sha, write
import scenes
import build_native

_saved=list(sys.path)
try:
    sys.path.insert(0,str(LAB/'experiments/global-pressure-v1'))
    oracle=load_module('water_independent_mac_oracle',LAB/'experiments/global-pressure-v1/verify_capture.py')
    legacy=load_module('water_capture_stream_owner',LAB/'experiments/global-pressure-v1/capture.py')
finally:sys.path[:]=_saved

FIELDS={'base':('<f8','cell'),'rhs':('<f8','cell'),'pressure':('<f8','cell'),
        'native_input_residual':('<f8','cell'),'tracer':('<f4','cell'),
        **{prefix+'_'+axis:('<f4','face_'+axis) for prefix in ('pre','corrected') for axis in 'uvw'}}


def geometry(row,config):
    h=1/row['cells_per_metre']
    return {'counts_xyz':row['counts_xyz'],'extents_m':[n*h for n in row['counts_xyz']],
            'origin_m':[0.0]*3,'density_kg_m3':config['material']['density_kg_m3'],
            'dynamic_viscosity_pa_s':config['material']['dynamic_viscosity_pa_s'],
            'boundary':'sealed_impermeable_free_slip','units':'SI'}


def shape(name,counts):
    result=list(counts)
    if name.startswith(('pre_','corrected_')):result['uvw'.index(name[-1])]+=1
    return result


def check_frame(root,geom,frame,dt):
    counts=geom['counts_xyz'];h=[geom['extents_m'][d]/counts[d] for d in range(3)];rho=geom['density_kg_m3']
    require(h[0]==h[1]==h[2],'cubic_cells','one scalar physical spacing')
    values={}
    for name,declaration in frame['arrays'].items():
        raw=read(root/declaration['path'],MIB);dtype=FIELDS[name][0]
        require(sha(raw)==declaration['sha256'] and declaration['shape_xyz']==shape(name,counts)
                and declaration['dtype']==dtype and len(raw)==math.prod(shape(name,counts))*np.dtype(dtype).itemsize,
                'raw_lineage',name)
        value=np.frombuffer(raw,dtype=dtype);require(bool(np.isfinite(value).all()),'nonfinite',name);values[name]=value.tolist()
    require(set(values)==set(FIELDS) and values['pressure'][0]==values['base'][0]==0,'gauge','complete gauge-fixed snapshot')
    pre=[values['pre_'+a] for a in 'uvw'];corrected=[values['corrected_'+a] for a in 'uvw']
    rebuilt=oracle.reconstruct_rhs(pre,counts,h,dt)
    require(np.allclose(rebuilt,values['rhs'],rtol=2e-14,atol=1e-14),'independent_rhs','same signed integrated face flux')
    product=oracle.full_action(values['pressure'],counts,h,rho)
    residual=[b-a for b,a in zip(values['rhs'],product)]
    residual_l2=math.sqrt(math.fsum(v*v for v in residual));rhs_l2=math.sqrt(math.fsum(v*v for v in values['rhs']))
    require(residual_l2<=max(1e-12,1e-8*rhs_l2),'reference_residual','unchanged reference gate')
    require(dt*max(map(abs,residual))/math.prod(h)<=1e-7,'predicted_divergence','unchanged gate')
    base_action=oracle.full_action(values['base'],counts,h,rho)
    require(np.allclose(np.asarray(values['rhs'])-base_action,values['native_input_residual'],rtol=2e-12,atol=2e-14),'input_residual','independent full operator on previous pressure')
    divergence=oracle.divergence(corrected,counts,h)
    actual=max(map(abs,divergence));require(actual<=1e-5,'actual_divergence','stored f32 correction')
    net=math.fsum(divergence)*math.prod(h);absolute=math.fsum(map(abs,divergence))*math.prod(h)
    require(abs(net)<=max(1e-12,1e-8*absolute),'net_flux','sealed volume conservation')
    for d,axis in enumerate('uvw'):
        dims=shape('corrected_'+axis,counts);array=np.asarray(values['corrected_'+axis]).reshape(dims[::-1])
        require(bool((np.take(array,[0,dims[d]-1],axis=2-d)==0).all()),'wall_flux','exact zero normal flow')
    courant=dt*math.fsum(max(map(abs,corrected[d]))/h[d] for d in range(3));require(courant<=.5,'courant','same gate')
    mu=frame['viscosity'];require(mu['dynamic_viscosity_pa_s']==geom['dynamic_viscosity_pa_s']
        and mu['kinematic_viscosity_m2_s']==geom['dynamic_viscosity_pa_s']/rho
        and mu['kinetic_after']<=mu['kinetic_before']*(1+1e-12)+1e-15,'viscosity','native material and dissipation')
    return {'reference_relative_residual':residual_l2/rhs_l2 if rhs_l2 else 0,'actual_divergence_max_s_inv':actual,'net_flux_m3_s':net,'courant':courant}


def fingerprint(geometry,entry,frames):
    return sha(canonical({'geometry':geometry,'force':entry['force'],'frames':[{n:d['sha256'] for n,d in f['arrays'].items()} for f in frames]}))


def generate(config,binary,output,fixture=False,observer=None,qualified_fixture=None,limit=None):
    reservation=scenes.reserve(config);config=reservation['configuration'];native=build_native.admit(binary)
    require(native['identity']['reservation_sha256']==sha(canonical(reservation)),'native_reservation','same measured scene reservation')
    if not fixture:
        require(qualified_fixture is not None,'fixture_required','qualify one training frame first')
        evidence=verify(qualified_fixture)
        require(decode(read(Path(qualified_fixture)/'manifest.json',4*MIB))['native_build_receipt']==native,'fixture_native','same qualified native binary and source closure')
        require(evidence['status']=='verified_fixture' and evidence['reservation_sha256']==sha(canonical(reservation)),'fixture_identity','same reservation')
    output=fresh(output);rows=reservation['scenes']
    if fixture:rows=[next(r for r in rows if r['split']=='train')]
    elif limit is not None:rows=[r for r in rows if r['split']=='train'][:limit]
    manifest={'schema':'rheon-sealed-water-capture-v1','fixture':fixture,'pilot':limit is not None,'reservation':reservation,
              'reservation_sha256':sha(canonical(reservation)),'native_build_receipt':native,'trajectories':[],'status':'candidate'}
    write(output/'reservation.json',reservation);payload=0;started=time.monotonic()
    for number,row in enumerate(rows):
        tid=row['trajectory_id'];directory=output/tid;directory.mkdir();geom=geometry(row,config)
        trajectory={'trajectory_id':tid,'entry':row,'geometry':geom,'frames':[]}
        selected=[0] if fixture else config['snapshot_steps']
        with (directory/'producer.stderr.log').open('xb') as stderr:
            with legacy.bounded_process([str(binary),'--produce'],120,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                    env={**os.environ,'RAYON_NUM_THREADS':'1'}) as process:
                process.stdin.write(canonical({'trajectory_id':tid,'fixture':fixture}));process.stdin.close()
                for step in selected:
                    size=struct.unpack('<I',legacy.read_exact(process.stdout,4))[0];require(0<size<=65536,'header','bounded snapshot header')
                    frame=decode(legacy.read_exact(process.stdout,size));require(frame['kind']=='frame' and frame['index']==step,'frame_order','reserved snapshots')
                    inventory={}
                    for descriptor in frame.pop('arrays'):
                        name=descriptor.pop('name');require(name in FIELDS and name not in inventory,'field_inventory','closed unique fields')
                        expected=math.prod(shape(name,geom['counts_xyz']))*np.dtype(FIELDS[name][0]).itemsize
                        require(descriptor['bytes']==expected and 0<expected<=MIB,'array_bound',name)
                        raw=legacy.read_exact(process.stdout,expected);payload+=len(raw);require(payload<=512*MIB,'capture_budget','bounded retained snapshots')
                        path=directory/f'frame-{step:03d}-{name}.bin';path.write_bytes(raw)
                        inventory[name]={**descriptor,'path':path.relative_to(output).as_posix(),'sha256':sha(raw)}
                    frame['arrays']=inventory;frame['independent']=check_frame(output,geom,frame,config['dt_s']);trajectory['frames'].append(frame)
                    if observer and row['split']!='reserved_test':
                        pressure=np.frombuffer(read(output/inventory['pressure']['path']),dtype='<f8')
                        preview=observer.frame(output,{'trajectory_id':tid,'counts_xyz':geom['counts_xyz'],'extents_m':geom['extents_m'],'dt_s':config['dt_s']},step,pressure=pressure,
                            pressure_label='Water reference pressure (Pa); independent physical checks passed; passive tracer')
                        observer.publish('Water reference generation',f'{number+1}/{len(rows)} scenes; {tid}; {geom["counts_xyz"]}; actual native step {step+1}',number+1,len(rows),preview=preview)
                size=struct.unpack('<I',legacy.read_exact(process.stdout,4))[0];require(0<size<=65536,'header','bounded completion')
                completed=decode(legacy.read_exact(process.stdout,size));expected_steps=1 if fixture else config['steps']
                require(completed=={'kind':'complete','steps':expected_steps,'snapshots':len(selected),'native_steps_qualified':expected_steps},'native_complete','every native step qualified')
                require(process.stdout.read(1)==b'' and process.wait(timeout=10)==0,'producer_exit','clean drained producer')
        trajectory['fingerprint']=fingerprint(geom,row,trajectory['frames'])
        manifest['trajectories'].append(trajectory)
        # Keep a recoverable candidate receipt after each independently checked scene.
        (output/'manifest.json').write_bytes(canonical(manifest))
    manifest['status']='verified_fixture' if fixture else 'verified_pilot' if limit is not None else 'verified_development'
    manifest.update(payload_bytes=payload,elapsed_s=time.monotonic()-started)
    (output/'manifest.json').write_bytes(canonical(manifest));return verify(output)


def verify(root):
    root=Path(root);raw=read(root/'manifest.json',4*MIB);manifest=decode(raw);reservation=scenes.validate_reservation(manifest['reservation'])
    require(manifest['reservation_sha256']==sha(canonical(reservation)) and manifest['status'] in ('verified_fixture','verified_pilot','verified_development'),'capture','accepted capture identity')
    expected=reservation['scenes']
    if manifest['fixture']:expected=[next(r for r in expected if r['split']=='train')]
    elif manifest['pilot']:expected=[r for r in expected if r['split']=='train'][:len(manifest['trajectories'])]
    require([t['entry'] for t in manifest['trajectories']]==expected,'population','complete ordered reserved parents')
    seen=set();snapshots=0
    for trajectory in manifest['trajectories']:
        geom=geometry(trajectory['entry'],reservation['configuration']);require(trajectory['geometry']==geom,'geometry','exact cubic water material')
        require(trajectory['fingerprint']==fingerprint(geom,trajectory['entry'],trajectory['frames']) and trajectory['fingerprint'] not in seen,'duplicate_scene','unique physical/numeric parents');seen.add(trajectory['fingerprint'])
        wanted=[0] if manifest['fixture'] else reservation['configuration']['snapshot_steps']
        require([f['index'] for f in trajectory['frames']]==wanted,'snapshot_population','complete ordered selected frames')
        for frame in trajectory['frames']:check_frame(root,geom,frame,reservation['configuration']['dt_s']);snapshots+=1
    return {'status':manifest['status'],'reservation_sha256':manifest['reservation_sha256'],'manifest_sha256':sha(raw),'parents':len(expected),'snapshots':snapshots,'payload_bytes':manifest.get('payload_bytes',0)}
