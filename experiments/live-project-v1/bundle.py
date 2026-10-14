#!/usr/bin/env python3
"""Lossless Tuldok field bundle from one independently verified native trajectory."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

PROFILE = Path(__file__).resolve().parents[2]/'experiments/local-pressure-v1'
sys.path.insert(0,str(PROFILE))
import capture
import contract

FIELDS = [('pressure','<f8','cell','Pa','accepted pressure'),
          ('tracer','<f4','cell','dimensionless','accepted transported tracer'),
          ('corrected_u','<f4','face_x','m/s','accepted projected velocity'),
          ('corrected_v','<f4','face_y','m/s','accepted projected velocity'),
          ('corrected_w','<f4','face_z','m/s','accepted projected velocity')]


def export(root, trajectory_id, tuldok, output, verify_capture=capture.verify, namespace='rheon-local-pressure-development'):
    verified=verify_capture(root)
    if verified['status']!='verified_development':raise ValueError('Complete independently verified development capture required')
    manifest_raw=contract.read(root/'manifest.json',4*contract.MIB)
    native=contract.decode(manifest_raw)
    trajectory=next(t for t in native['trajectories'] if t['trajectory_id']==trajectory_id)
    sys.path.insert(0,str(tuldok))
    from adapters.rheon.generate import npy
    import scientific_fields
    geometry=trajectory['geometry'];counts=geometry['counts_xyz']
    members={}
    manifest={'schema_version':1,'format':'scientific_field_sequence_v1','grid':{'shape_xyz':counts,
              'origin_xyz':geometry['origin_m'],'spacing_xyz':[geometry['extents_m'][d]/counts[d] for d in range(3)],
              'orientation':[[1,0,0],[0,1,0],[0,0,1]],'units':'m'},'time_units':'s',
              'fields':[{'id':name,'label':name.replace('_',' '),'dtype':dtype,'centering':center,'units':units,'stage':stage}
                        for name,dtype,center,units,stage in FIELDS],
              'frames':[],'family':{'namespace':namespace,'id':trajectory_id},'sources':[],
              'provenance':{'producer':'Verified native capture; external live integration inspection','source_id':contract.sha(manifest_raw)}}
    for frame in trajectory['frames']:
        row={'time':str(frame['time_s']),'arrays':[]}
        for name,dtype,center,_units,_stage in FIELDS:
            source=frame['arrays'][name];raw=contract.read(root/source['path'],contract.MIB)
            if contract.sha(raw)!=source['sha256']:raise ValueError('Raw field drift')
            path=f'arrays/{frame["index"]:03d}-{name}.npy'
            data=npy(raw,dtype,source['shape_xyz'][::-1]);members[path]=data
            row['arrays'].append({'field':name,'path':path,'bytes':len(data),'sha256':contract.sha(data)})
        manifest['frames'].append(row)
    for path,data,description in [('capture-manifest.json',manifest_raw,'Original measured capture manifest'),
                                  ('independent-verification.json',contract.read(root/'verification.json',4*contract.MIB),'Original independent physics verification')]:
        members[path]=data
        manifest['sources'].append({'path':path,'bytes':len(data),'sha256':contract.sha(data),'description':description})
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_STORED) as archive:
        archive.writestr(zipfile.ZipInfo('manifest.json'),contract.canonical(manifest))
        for path,data in members.items():archive.writestr(zipfile.ZipInfo(path),data)
    data=stream.getvalue()
    scientific_fields.parse_bundle(data)
    with output.open('xb') as file:file.write(data)
    return {'status':'verified_lossless_field_bundle','trajectory_id':trajectory_id,'frames':len(trajectory['frames']),
            'fields':len(FIELDS),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--trajectory',required=True)
    parser.add_argument('--tuldok-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capture-profile',choices=('local','cubic'),default='local')
    args=parser.parse_args()
    verifier=capture.verify;namespace='rheon-local-pressure-development'
    if args.capture_profile=='cubic':
        profile=PROFILE.parent/'cubic-pressure-v1'
        sys.path.insert(0,str(profile))
        verifier=contract.load_module('cubic_bundle_capture',profile/'cubic_capture.py').verify
        namespace='rheon-cubic-pressure-development'
    print(json.dumps(export(args.capture,args.trajectory,args.tuldok_source,args.output,verifier,namespace)))


if __name__=='__main__':main()
