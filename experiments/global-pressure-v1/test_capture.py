#!/usr/bin/env python3
"""Meaningful literal/dyadic oracles and isolated contract corruptions.

Unit fixture is analytic zero-flow, not an independently evolved reference.
Real process/native/source/physics closure is separately run by main integration.
"""
import json
import copy
from fractions import Fraction
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import verify_capture as v
import capture as c

def fixture(root):
    reservation=json.loads(v.RESERVATION.read_bytes())
    e=reservation['presets']['smoke']['trajectories'][0]
    tid=e['trajectory_id'];(root/tid).mkdir()
    geometry={'counts_xyz':[16]*3,'extents_m':[1,1,1],'origin_m':[0.0]*3,'density_kg_m3':1.25,'boundary':'sealed_zero_normal_mac','units':'SI'}
    def payload(name,values,dtype,dims,units):
        data=b''.join(struct.pack('<d' if dtype=='<f8' else '<f',x) for x in values)
        path=tid+'/'+name+'.bin';(root/path).write_bytes(data)
        return {'path':path,'shape_xyz':dims,'dtype':dtype,'units':units,'layout':'x-fastest','bytes':len(data),'sha256':v.digest(data)}
    # Formula is literal accepted initial tracer authority; no producer import.
    tracer=[v.f32(.25*math.exp(-sum(((a+.5)/16-.5)**2 for a in (x,y,z))/(2*.08**2))) for z in range(16) for y in range(16) for x in range(16)]
    initial={'pressure':payload('initial-pressure',[0]*4096,'<f8',[16]*3,'Pa'),'tracer':payload('initial-tracer',tracer,'<f4',[16]*3,'dimensionless')}
    for d,n in enumerate(('u','v','w')):initial[n]=payload('initial-'+n,[0]*math.prod(v.shape(d)),'<f4',v.shape(d),'m/s')
    arrays={'rhs':payload('rhs',[0]*4096,'<f8',[16]*3,'m3/s2'),'pressure':payload('pressure',[0]*4096,'<f8',[16]*3,'Pa'),'tracer':payload('tracer',tracer,'<f4',[16]*3,'dimensionless')}
    for d,n in enumerate(('u','v','w')):
        for prefix in ('pre_','corrected_'):arrays[prefix+n]=payload(prefix+n,[0]*math.prod(v.shape(d)),'<f4',v.shape(d),'m/s')
    frame={'index':0,'time_s':e['dt_s'],'forcing_time_s':0.0,'previous_pressure':initial['pressure']['path'],'previous_state':{n:initial[n]['path'] for n in ('u','v','w','tracer')},'arrays':arrays}
    return {'schema':'rheon-global-pressure-v1','status':'fixture_complete','fixture':True,'preset':'smoke','selection':{'split':'train','trajectory_id':tid},'counts_xyz':[16]*3,'density_kg_m3':1.25,'reservation_sha256':v.digest(v.RESERVATION.read_bytes()),'config_sha256':v.digest(v.CONFIG.read_bytes()),'provenance':{'native_receipt':{'base_commit':v.BASE}},'trajectories':[{'trajectory_id':tid,'reservation_entry':e,'geometry':geometry,'geometry_sha256':v.digest(v.canonical(geometry)),'initial':initial,'frames':[frame]}]}

class HandPhysics(unittest.TestCase):
    def setUp(self):
        self.counts=[2,2,1];self.h=[.5,.25,2.];self.rho=2.;self.dt=.125;self.p=[0.,1.,2.,3.]
        self.pre=[[0.]*math.prod(v.shape(d,self.counts)) for d in range(3)]
        # Literal dyadic face gradients: x=.125, y=.5, z has no interior.
        self.pre[0][1]=.125;self.pre[0][4]=.125
        self.pre[1][2]=.5;self.pre[1][3]=.5
    def test_literal_anisotropic_full_stencil_includes_gauge_row(self):
        self.assertEqual(v.full_action(self.p,self.counts,self.h,self.rho),[-4.5,-3.5,3.5,4.5])
    def test_dyadic_flux_rhs_and_f32_gradient_all_axes_walls(self):
        rhs=v.reconstruct_rhs(self.pre,self.counts,self.h,self.dt)
        self.assertEqual(rhs,[-4.5,-3.5,3.5,4.5])
        corrected=v.corrected_velocity(self.pre,self.p,self.counts,self.h,self.rho,self.dt)
        self.assertTrue(all(x==0 for field in corrected for x in field))
        self.assertEqual(v.physics(self.pre,rhs,self.p,corrected,self.counts,self.h,self.rho,self.dt)['residual_l2'],0)
    def test_rhs_sign_error_is_distinct_from_solver_gate(self):
        rhs=[4.5,3.5,-3.5,-4.5]
        corrected=v.corrected_velocity(self.pre,self.p,self.counts,self.h,self.rho,self.dt)
        with self.assertRaises(v.CaptureError) as error:v.physics(self.pre,rhs,self.p,corrected,self.counts,self.h,self.rho,self.dt)
        self.assertEqual(error.exception.code,'rhs_reconstruction')
    def test_cell_zero_residual_is_not_discarded(self):
        self.assertEqual(v.full_action(self.p,self.counts,self.h,self.rho)[0],-4.5)
        guessed=self.p.copy();guessed[1]+=.25
        with self.assertRaises(v.CaptureError) as error:v.physics(self.pre,[-4.5,-3.5,3.5,4.5],guessed,v.corrected_velocity(self.pre,guessed,self.counts,self.h,self.rho,self.dt),self.counts,self.h,self.rho,self.dt)
        self.assertEqual(error.exception.code,'reference_residual')
    def test_binary32_rounding_is_staged_once(self):
        p=[0.,1./3.,2./3.,1.]
        out=v.corrected_velocity(self.pre,p,self.counts,self.h,self.rho,self.dt)
        self.assertEqual(struct.pack('<f',out[0][1]),struct.pack('<f',.125-.125*(1/(2*.5))*(1/3)))
    def test_literal_3d_anisotropic_all_axes(self):
        counts=[2,2,2];h=[.5,.25,2.];p=[0.,1.,2.,3.,4.,5.,6.,7.]
        pre=[[0.]*12 for _ in range(3)]
        for face in [1,4,7,10]:pre[0][face]=.125
        for face in [2,3,8,9]:pre[1][face]=.5
        for face in [4,5,6,7]:pre[2][face]=.125
        literal=[-4.625,-3.625,3.375,4.375,-4.375,-3.375,3.625,4.625]
        self.assertEqual(v.full_action(p,counts,h,2.),literal)
        self.assertEqual(v.reconstruct_rhs(pre,counts,h,.125),literal)
        corrected=v.corrected_velocity(pre,p,counts,h,2.,.125)
        self.assertTrue(all(value==0. for field in corrected for value in field))
        self.assertEqual(v.physics(pre,literal,p,corrected,counts,h,2.,.125)['actual_divergence_max_s_inv'],0.)
    def test_predicted_divergence_gate_isolated_after_residual_passes(self):
        counts=[2,1,1];h=[2**-20]*3;dt=.125;p=[0.,2**-40]
        pre=[[0.]*math.prod(v.shape(d,counts)) for d in range(3)]
        corrected=v.corrected_velocity(pre,p,counts,h,1.,dt)
        with self.assertRaises(v.CaptureError) as error:v.physics(pre,[0.,0.],p,corrected,counts,h,1.,dt)
        self.assertEqual(error.exception.code,'predicted_divergence')
    def test_actual_binary32_divergence_gate_isolated(self):
        # Exact rational gauge solve of integer graph Lp=scatter(16*pre):
        # edges(0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5).
        counts=[3,2,1];h=[2**-16]*3;dt=2**-20
        pre=[[0.]*math.prod(v.shape(d,counts)) for d in range(3)]
        for face,value in zip([1,2,5,6],[.8841948509216309,-.36847081780433655,-.8262166976928711,-.3446801006793976]):pre[0][face]=value
        for face,value in zip([3,4,5],[.7309670448303223,-.2294968217611313,.523900032043457]):pre[1][face]=value
        rhs=[-.0003943266347050667,.00036185607677907683,-.00021786397701362148,.0003801718121394515,-.00017359214325551875,.000043754866055678576]
        p=[0.,float(Fraction(639784093,62914560)),float(Fraction(2268893,6291456)),float(Fraction(986091107,62914560)),float(Fraction(202334867,31457280)),float(Fraction(101294559,20971520))]
        corrected=v.corrected_velocity(pre,p,counts,h,1.,dt)
        with self.assertRaises(v.CaptureError) as error:v.physics(pre,rhs,p,corrected,counts,h,1.,dt)
        self.assertEqual(error.exception.code,'actual_divergence')
    def test_sealed_cosine_action_eigenvalue(self):
        counts=[4,3,2];h=[.25,.5,1.];rho=1.25
        mode=[math.cos(math.pi*(x+.5)/4)*math.cos(2*math.pi*(y+.5)/3)*math.cos(math.pi*(z+.5)/2) for z in range(2) for y in range(3) for x in range(4)]
        eigenvalue=sum((math.prod(h[a] for a in range(3) if a!=d)/(rho*h[d]))*2*(1-math.cos(math.pi*k/counts[d])) for d,k in enumerate([1,2,1]))
        for got,p in zip(v.full_action(mode,counts,h,rho),mode):self.assertAlmostEqual(got,eigenvalue*p,places=13)

class ContractMutations(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.manifest=fixture(self.root)
        self.assertEqual(v.verify(self.root,self.manifest,check_sources=False)['frames'],1)
    def assert_code(self,code):
        with self.assertRaises(v.CaptureError) as error:v.verify(self.root,self.manifest,check_sources=False)
        self.assertEqual(error.exception.code,code)
    def array(self):return self.manifest['trajectories'][0]['frames'][0]['arrays']['rhs']
    def mutate_payload(self,desc,offset,value,rebind_hash=False):
        path=self.root/desc['path'];data=bytearray(path.read_bytes());struct.pack_into('<d' if desc['dtype']=='<f8' else '<f',data,offset,value);path.write_bytes(data)
        if rebind_hash:desc['sha256']=v.digest(data)
    def test_byte_corruption_hits_payload_hash(self):self.mutate_payload(self.array(),0,1.);self.assert_code('payload_hash')
    def test_rehashed_rhs_corruption_hits_rhs_reconstruction(self):self.mutate_payload(self.array(),0,1.,True);self.assert_code('rhs_reconstruction')
    def test_rehashed_nonfinite_hits_finite_boundary(self):self.mutate_payload(self.array(),0,float('nan'),True);self.assert_code('array_nonfinite')
    def test_shape_before_allocation(self):self.array()['shape_xyz']=[16,16,15];self.assert_code('array_shape')
    def test_units_are_explicit(self):self.array()['units']='Pa';self.assert_code('array_format')
    def test_final_identity_rejected_before_array_access(self):self.manifest['trajectories'][0]['trajectory_id']='smoke-trajectory_test-100200';self.assert_code('frame_coverage')
    def test_history_reference_is_actual_previous_pressure(self):self.manifest['trajectories'][0]['frames'][0]['previous_pressure']='bogus.bin';self.assert_code('history_lineage')
    def test_time_corruption(self):self.manifest['trajectories'][0]['frames'][0]['time_s']+=1.;self.assert_code('frame_time')
    def test_unsafe_payload_path(self):self.array()['path']='../outside.bin';self.assert_code('unsafe_path')
    def test_symlink_payload(self):
        desc=self.array();path=self.root/desc['path'];target=self.root/'copy';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target);self.assert_code('unsafe_path')
    def test_wall_velocity_corruption_rehashed_hits_boundary(self):
        desc=self.manifest['trajectories'][0]['frames'][0]['arrays']['pre_u'];self.mutate_payload(desc,0,.25,True);self.assert_code('boundary_flux')
    def test_rehashed_pressure_shift_hits_gauge(self):
        desc=self.manifest['trajectories'][0]['frames'][0]['arrays']['pressure'];self.mutate_payload(desc,0,1.,True);self.assert_code('pressure_gauge')
    def test_rehashed_corrected_face_hits_f32_correction(self):
        desc=self.manifest['trajectories'][0]['frames'][0]['arrays']['corrected_u'];self.mutate_payload(desc,4,.25,True);self.assert_code('f32_correction')
    def test_signed_zero_rehashed_hits_bitwise_correction(self):
        desc=self.manifest['trajectories'][0]['frames'][0]['arrays']['corrected_u']
        self.mutate_payload(desc,4,-0.0,True)
        self.assert_code('f32_correction')
    def test_complete_zero_history_rejected_without_tolerance(self):
        trajectory=self.manifest['trajectories'][0];initial=trajectory['initial'];original=trajectory['frames'][0]
        self.manifest['fixture']=False;self.manifest['status']='development_complete'
        frames=[];previous_pressure=initial['pressure']['path'];previous_state={n:initial[n]['path'] for n in ('u','v','w','tracer')}
        for i in range(24):
            frame=copy.deepcopy(original);frame.update(index=i,time_s=(i+1)*trajectory['reservation_entry']['dt_s'],forcing_time_s=i*trajectory['reservation_entry']['dt_s'],previous_pressure=previous_pressure,previous_state=previous_state)
            for name,desc in frame['arrays'].items():
                old=self.root/desc['path'];new=old.with_name(f'frame{i:03d}-{name}.bin');new.write_bytes(old.read_bytes());desc['path']=str(new.relative_to(self.root))
            frames.append(frame);previous_pressure=frame['arrays']['pressure']['path'];previous_state={n:frame['arrays']['corrected_'+n]['path'] for n in ('u','v','w')};previous_state['tracer']=frame['arrays']['tracer']['path']
        for desc in original['arrays'].values():(self.root/desc['path']).unlink()
        trajectory['frames']=frames
        self.assert_code('nontrivial_history')
    def test_raw_whole_history_cross_split_identity_ignores_seed_id_paths(self):
        first=self.manifest['trajectories'][0];second=copy.deepcopy(first)
        second['trajectory_id']='smoke-validation-100100';second['reservation_entry']['seed']=100100;second['reservation_entry']['split']='validation'
        for inventory in [second['initial']]+[f['arrays'] for f in second['frames']]:
            for desc in inventory.values():desc['path']='renamed/'+Path(desc['path']).name
        a=v.trajectory_fingerprint(first['geometry'],first['initial'],first['frames']);b=v.trajectory_fingerprint(second['geometry'],second['initial'],second['frames'])
        self.assertEqual(a,b)
        seen={};v.admit_lineage(seen,a,'train',first['trajectory_id'])
        with self.assertRaises(v.CaptureError) as error:v.admit_lineage(seen,b,'validation',second['trajectory_id'])
        self.assertEqual(error.exception.code,'cross_split_duplicate')
    def test_extra_payload_not_silently_ignored(self):(self.root/'unlisted.bin').write_bytes(b'x');self.assert_code('array_inventory')

    def test_physical_fingerprint_numeric_and_signed_zero_aliases_rejected_as_dupe(self):
        first=self.manifest['trajectories'][0];alternate=copy.deepcopy(first['geometry'])
        alternate['extents_m']=[1.0]*3;alternate['origin_m']=[-0.0]*3
        self.assertEqual(first['geometry'],alternate)
        self.assertNotEqual(v.digest(v.canonical(first['geometry'])),v.digest(v.canonical(alternate)))
        a=v.trajectory_fingerprint(first['geometry'],first['initial'],first['frames'])
        b=v.trajectory_fingerprint(alternate,first['initial'],first['frames'])
        self.assertEqual(a,b)
        seen={};v.admit_lineage(seen,a,'train','train')
        with self.assertRaises(v.CaptureError) as error:v.admit_lineage(seen,b,'validation','validation')
        self.assertEqual(error.exception.code,'cross_split_duplicate')
    def test_raw_geometry_hash_covers_validated_actual_numeric_representation(self):
        geometry=self.manifest['trajectories'][0]['geometry']
        geometry['extents_m']=[1.0]*3;geometry['origin_m']=[-0.0]*3
        self.manifest['trajectories'][0]['geometry_sha256']=v.digest(v.canonical(geometry))
        self.assertEqual(v.verify(self.root,self.manifest,check_sources=False)['frames'],1)
    def test_geometry_bool_numbers_rejected_before_hash_identity(self):
        geometry=self.manifest['trajectories'][0]['geometry']
        geometry['extents_m']=[True]*3;geometry['origin_m']=[False]*3
        # Retain the original valid numeric geometry hash to isolate type boundary.
        self.assert_code('geometry_type')
    def test_reservation_integer_bool_is_rejected(self):
        self.manifest['trajectories'][0]['reservation_entry']['seed']=True
        self.assert_code('reservation_type')
    def test_native_iterations_bool_is_rejected(self):
        self.manifest['trajectories'][0]['frames'][0]['iterations']=False
        self.assert_code('native_diagnostic_type')
    def test_native_metric_string_is_rejected(self):
        metrics={'residual_l2':0.,'residual_max':0.,'rhs_l2':0.,'relative_residual':None,'predicted_divergence_max_s_inv':0.,'actual_divergence_max_s_inv':0.,'actual_divergence_rms_s_inv':0.,'net_flux_m3_s':0.,'courant':0.}
        metrics['courant']='0.0'
        self.manifest['trajectories'][0]['frames'][0]['metrics']=metrics
        self.assert_code('native_diagnostic_type')
    def test_native_sgs_diagnostic_array_is_rejected(self):
        metrics={'residual_l2':0.,'residual_max':0.,'rhs_l2':0.,'relative_residual':None,'predicted_divergence_max_s_inv':0.,'actual_divergence_max_s_inv':0.,'actual_divergence_rms_s_inv':0.,'net_flux_m3_s':0.,'courant':0.}
        metrics['sgs']={'iterations':0,'gates':dict(metrics),'pressure_difference_l2':[]}
        self.manifest['trajectories'][0]['frames'][0]['metrics']=metrics
        self.assert_code('native_diagnostic_type')
    def test_unknown_manifest_field_is_rejected(self):self.manifest['ignored_field']=123;self.assert_code('schema_keys')
    def test_boolean_frame_index_is_rejected(self):self.manifest['trajectories'][0]['frames'][0]['index']=False;self.assert_code('schema_type')
    def test_unknown_array_field_is_rejected(self):self.array()['ignored_field']=123;self.assert_code('schema_keys')
    def test_complete_source_inventory_cannot_be_empty(self):
        from prepare_dependency import load_contract, receipt_for
        dependency, _, _, manifest_hash=load_contract()
        self.manifest['provenance']={'native_receipt':receipt_for(dependency,manifest_hash),'source_sha256':{},'prepared_source':'/absent','binary_path':'/absent','binary_sha256':'0'*64,'build_receipt_sha256':'0'*64,'lab_commit':'0'*40}
        with self.assertRaises(v.CaptureError) as error:v.verify(self.root,self.manifest)
        self.assertEqual(error.exception.code,'source_inventory')

class NativeProducerBoundary(unittest.TestCase):
    def test_invalid_native_request_never_emits_payload(self):
        binary=os.environ.get('RHEON_TEST_BINARY')
        if not binary:self.skipTest('requires qualified real native binary')
        valid={'preset':'smoke','trajectory_id':'smoke-train-100000','frames':1,'fixture':True}
        cases=[]
        for field,value in [('preset','full'),('trajectory_id','smoke-trajectory_test-100200'),('frames',True),('fixture','true'),('unknown',1)]:
            request=dict(valid);request[field]=value;cases.append(json.dumps(request))
        cases.append('{"preset":"smoke","preset":"full","trajectory_id":"smoke-train-100000","frames":1,"fixture":true}')
        for request in cases:
            result=subprocess.run([binary,'--produce'],input=request,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,1);self.assertEqual(result.stdout,'');self.assertIn('capture_failure:',result.stderr)

class Admission(unittest.TestCase):
    def test_full_and_test_cli_fail_without_output(self):
        for preset,split in [('full','train'),('smoke','trajectory_test'),('smoke','geometry_test'),('smoke','regime_test')]:
            with tempfile.TemporaryDirectory() as t:
                output=Path(t)/'should-not-exist'
                p=subprocess.run([sys.executable,str(HERE/'capture.py'),'generate','--preset',preset,'--split',split,'--binary','/absent','--prepared-source','/absent','--output',str(output)],capture_output=True,text=True)
                self.assertEqual(p.returncode,1);self.assertIn('development_admission',p.stderr);self.assertFalse(output.exists())
    def test_explicit_invalid_or_validation_fixture_refused_before_binary_output(self):
        for split,tid,frames in [('train','smoke-train-100000','0'),('train','smoke-train-100000','-1'),('train','smoke-train-100000','25'),('validation','smoke-validation-100100','1')]:
            with tempfile.TemporaryDirectory() as t:
                output=Path(t)/'should-not-exist'
                p=subprocess.run([sys.executable,str(HERE/'capture.py'),'generate','--preset','smoke','--split',split,'--trajectory',tid,'--fixture-frames',frames,'--binary','/absent','--prepared-source','/absent','--output',str(output)],capture_output=True,text=True)
                self.assertEqual(p.returncode,1);self.assertIn('development_admission',p.stderr);self.assertFalse(output.exists())
    def test_duplicate_json_keys_and_nonfinite_are_typed(self):
        for data,code in [('{"fixture":true,"fixture":false}','duplicate_json_key'),('{"time":NaN}','json_nonfinite')]:
            with self.assertRaises(v.CaptureError) as error:v.strict_json(data)
            self.assertEqual(error.exception.code,code)
    def test_stalled_child_stream_is_killed_with_bounded_timeout(self):
        started=time.monotonic()
        with self.assertRaises(ValueError) as error:
            with c.bounded_process([sys.executable,'-c','import time;time.sleep(60)'],.1,stdout=subprocess.PIPE) as process:
                process.stdout.read(1)
        self.assertIn('generation timeout',str(error.exception));self.assertLess(time.monotonic()-started,2)
        self.assertIsNotNone(process.poll())
    def test_alternate_compatible_native_source_rejected_before_build_output(self):
        from prepare_dependency import PreparationError
        original=c.LAB/'runs/global-pressure-v1-dependency/rheon'
        if not original.exists():self.skipTest('real prepared closure unavailable')
        import shutil
        with tempfile.TemporaryDirectory() as t:
            alternate=Path(t)/'rheon';shutil.copytree(original,alternate)
            pressure=alternate/'src/pressure.rs'
            # API still source-compatible, scientific implementation/source differs.
            pressure.chmod(0o644)
            pressure.write_bytes(pressure.read_bytes()+b'\n// alternate compatible source\n')
            output=c.LAB/'runs/global-pressure-v1-build/should-never-be-created-alternate'
            from types import SimpleNamespace
            with self.assertRaises(PreparationError) as error:c.build(SimpleNamespace(prepared_source=alternate,output=output,cargo=Path('/absent')))
            self.assertIn('content/mode inventory mismatch',str(error.exception));self.assertFalse(output.exists())
    def test_immutable_output_refuses_overwrite(self):
        parent=c.LAB/'runs/global-pressure-v1-capture';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as t:
            with self.assertRaises(FileExistsError):c.new_output(Path(t),'global-pressure-v1-capture')
    def test_unsafe_output_refused(self):
        with self.assertRaises(ValueError):c.new_output(Path('/tmp/invalid-capture'),'global-pressure-v1-capture')

if __name__=='__main__':unittest.main()
