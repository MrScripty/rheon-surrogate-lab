"""Same native research protocol, with bounded reads and owned child cleanup."""
import json
import os
import select
import subprocess
import time
import numpy as np
from native_client import NativeClient


class BoundedNativeClient(NativeClient):
    def __init__(self,*args,io_timeout_s=60.,**kwargs):
        self._incoming=bytearray();self.io_timeout_s=io_timeout_s;self.last_custody=None
        if not 0<io_timeout_s<=120:raise ValueError('Bounded native packet deadline required')
        try:super().__init__(*args,**kwargs)
        except BaseException:
            if hasattr(self,'child'):self._reap(False)
            raise

    def _pull(self,deadline):
        remaining=deadline-time.monotonic()
        if remaining<=0 or not select.select([self.child.stdout],[],[],remaining)[0]:
            raise TimeoutError('Native packet deadline exceeded')
        block=os.read(self.child.stdout.fileno(),65536)
        if not block:raise EOFError('Native protocol stream ended early')
        self._incoming.extend(block)

    def read_json(self):
        deadline=time.monotonic()+self.io_timeout_s
        while True:
            end=self._incoming.find(b'\n')
            if end>=0:
                if end+1>65536:raise EOFError('Native JSON packet exceeds bound')
                line=bytes(self._incoming[:end+1]);del self._incoming[:end+1]
                return json.loads(line)
            if len(self._incoming)>=65536:raise EOFError('Native JSON packet exceeds bound')
            self._pull(deadline)

    def read_array(self,count,dtype):
        size=count*np.dtype(dtype).itemsize
        if not 0<=size<=256**3*8:raise ValueError('Native array exceeds declared bound')
        deadline=time.monotonic()+self.io_timeout_s
        while len(self._incoming)<size:self._pull(deadline)
        value=np.frombuffer(bytes(self._incoming[:size]),dtype=dtype).copy()
        del self._incoming[:size]
        return value

    def _reap(self,graceful):
        if self.child.poll()is None:
            self.child.terminate()
            try:self.child.wait(timeout=2)
            except subprocess.TimeoutExpired:self.child.kill();self.child.wait(timeout=2)
        else:self.child.wait(timeout=2)
        for stream in (self.child.stdin,self.child.stdout):
            if stream is not None:
                try:stream.close()
                except (BrokenPipeError,OSError):pass
        if hasattr(self,'log'):self.log.close()
        self.last_custody=dict(pid=self.child.pid,returncode=self.child.returncode,
            reaped=self.child.poll()is not None,graceful_protocol_complete=graceful)
        return self.last_custody

    def close(self):
        if self.last_custody is not None:return self.last_custody
        complete=False;failure=None;previous=self.io_timeout_s;self.io_timeout_s=2.
        try:
            if self.child.poll()is None:
                self.write_json(dict(kind='stop'))
                if self.read_json()!={'kind':'complete'}:raise ValueError('Native completion packet differs')
                self.child.stdin.close();self.child.wait(timeout=2)
                if self._incoming:raise ValueError('Unexpected native trailing bytes')
                if os.read(self.child.stdout.fileno(),1)!=b'':raise ValueError('Unexpected native trailing output')
                complete=True
        except BaseException as error:failure=error
        finally:self.io_timeout_s=previous;custody=self._reap(complete)
        if failure is not None:raise failure
        return custody
