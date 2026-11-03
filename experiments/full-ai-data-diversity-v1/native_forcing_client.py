"""Owned reference client with an explicit, read-only forcing qualification seam."""
from support import BASE
from bounded_native_client import BoundedNativeClient
import numpy as np


class NativeForcingClient(BoundedNativeClient):
    def __init__(self,scene,directory,**kwargs):
        super().__init__(scene,directory,build=BASE/'native-forcing-build-02',**kwargs)

    def preview_force(self,step,velocity):
        self.write_json(dict(kind='forcing',step=step))
        for value in velocity:
            self.child.stdin.write(memoryview(np.ascontiguousarray(value,dtype='<f4')).cast('B'))
        self.child.stdin.flush()
        packet=self.read_json()
        if packet!={'kind':'forcing','step':step,'state_unchanged':True}:
            raise ValueError('Expected read-only native force preview')
        counts=self.scene['counts']
        return [self.read_array((counts[d]+1)*__import__('math').prod(counts[a]for a in range(3)if a!=d),'<f4')for d in range(3)]
