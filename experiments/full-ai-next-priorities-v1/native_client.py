"""Own one isolated native integrator; reference pressure solves stay explicit."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from study_support import ROOT, BASE, read, sha

FIELDS=('pressure','vx','vy','vz','tracer')

class NativeClient:
    def __init__(self,scene,directory,build=None):
        self.scene=scene
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.build=build or BASE/'temporal-build-02'
        receipt=read(self.build/'build-receipt.json')
        binary=self.build/'rheon-temporal-probe'
        if receipt['status']!='qualified_build' or sha(binary)!=receipt['binary_sha256']:
            raise ValueError('Qualified native binary required')
        self.log=(self.directory/'native-stderr.log').open('xb')
        self.child=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            env={**os.environ,'RAYON_NUM_THREADS':'1'},start_new_session=True)
        self.write_json(scene)
        self.ready=self.read_json()
        if self.ready['kind']!='ready':raise ValueError('Native initialization did not acknowledge')
        self.last_request=None
        self.last_frame=None

    def write_json(self,value):
        self.child.stdin.write((json.dumps(value,separators=(',',':'))+'\n').encode())
        self.child.stdin.flush()

    def read_json(self):
        line=self.child.stdout.readline(65537)
        if not line.endswith(b'\n') or len(line)>65536:
            raise EOFError('Native control packet truncated or oversized')
        return json.loads(line)

    def read_array(self,count,dtype):
        size=count*np.dtype(dtype).itemsize
        if size>256**3*8:raise ValueError('Native field size exceeds bound')
        blocks=[]
        while size:
            block=self.child.stdout.read(min(size,65536))
            if not block:raise EOFError('Native field truncated')
            blocks.append(block);size-=len(block)
        return np.frombuffer(b''.join(blocks),dtype=dtype).copy()

    def inspect(self):
        self.write_json(dict(kind='inspect'))
        packet=self.read_json()
        if packet['kind']!='state':raise ValueError('Expected native state')
        counts=self.scene['counts'];cells=math.prod(counts)
        expected=[(counts[d]+1)*math.prod(counts[a]for a in range(3)if a!=d)for d in range(3)]
        if packet['pressure_cells']!=cells or packet['tracer_cells']!=cells or packet['velocity_lengths']!=expected:
            raise ValueError('Staggered state shape mismatch')
        state={'pressure':self.read_array(cells,'<f8')}
        for name,n in zip(('vx','vy','vz'),expected):state[name]=self.read_array(n,'<f4')
        state['tracer']=self.read_array(cells,'<f4')
        if not all(np.isfinite(v).all()for v in state.values()):raise ValueError('Nonfinite native state')
        return packet,state

    def restore(self,step,state):
        self.write_json(dict(kind='restore',step=step))
        for name in FIELDS:
            array=np.ascontiguousarray(state[name],dtype='<f8'if name=='pressure'else'<f4')
            self.child.stdin.write(memoryview(array).cast('B'))
        self.child.stdin.flush()
        result=self.read_json()
        if result!={'kind':'restored','step':step,'numerical_steps_executed':0}:
            raise ValueError('Native state restoration not acknowledged')
        return result

    def advance(self,method='spectral',predict=None):
        import numerics
        self.write_json(dict(kind='advance',native=method=='native-pcg'))
        packet=self.read_json()
        if packet['kind']=='predict':
            cells=math.prod(self.scene['counts'])
            if packet['cells']!=cells:raise ValueError('Native request size mismatch')
            shape=(1,1,*self.scene['counts'][::-1])
            base=torch.from_numpy(self.read_array(cells,'<f8').reshape(shape))
            residual=torch.from_numpy(self.read_array(cells,'<f8').reshape(shape))
            h=torch.tensor([1/self.scene['cells_per_metre']],dtype=torch.float64)
            rho=torch.tensor([1000.],dtype=torch.float64)
            dt=torch.tensor([1/60],dtype=torch.float64)
            with torch.inference_mode():
                rhs=residual+numerics.fine_action(base,h,rho)
                sample=dict(base=base,native_input_residual=residual,rhs=rhs,cell_size_m=h,rho=rho,dt=dt)
                tick=time.perf_counter()
                pressure=base+numerics.correction(residual,h,rho,None) if method=='spectral'else predict(sample)
                pressure=pressure-pressure[:,:,:1,:1,:1].clone()
                predict_seconds=time.perf_counter()-tick
            array=np.ascontiguousarray(pressure.numpy(),dtype='<f8')
            self.child.stdin.write(memoryview(array).cast('B'));self.child.stdin.flush()
            self.last_request=dict(base=base.numpy().reshape(-1).copy(),rhs=rhs.numpy().reshape(-1).copy(),
                native_input_residual=residual.numpy().reshape(-1).copy(),predict_seconds=predict_seconds)
            packet=self.read_json()
        elif method!='native-pcg':raise ValueError('Expected external pressure request')
        if packet['kind']!='frame':raise ValueError('Expected native frame')
        self.last_frame=packet
        return packet

    def close(self):
        if self.child.poll()is None:
            try:
                self.write_json(dict(kind='stop'))
                if self.read_json()['kind']!='complete':raise ValueError('No native completion packet')
                self.child.stdin.close()
                if self.child.stdout.read(1)!=b'':raise ValueError('Unexpected native trailing output')
                self.child.wait(timeout=10)
            finally:
                if self.child.poll()is None:self.child.terminate();self.child.wait(timeout=10)
        self.log.close()
        return dict(pid=self.child.pid,returncode=self.child.returncode,reaped=self.child.poll()is not None)

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
