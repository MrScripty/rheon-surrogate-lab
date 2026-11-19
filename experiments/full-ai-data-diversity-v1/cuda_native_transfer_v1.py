"""Typed native-state transfers with pinned storage and explicit ownership.

Upload packs native bytes once; download queues every native field into one
independently owned pinned allocation and synchronizes the current stream once.
The adapter performs no learned or physical computation.
"""
import torch
FIELDS=('pressure','vx','vy','vz','tracer')

class UploadedState(dict):
    """Keep pinned H2D source and packed device storage alive with field views."""
    pass

def layout(rows,device):
    entries=[];offset=0;gaps=[]
    if not 1<=len(rows)<=4:raise ValueError('Complete one-to-four-state transfer required')
    for step,row in enumerate(rows):
        if set(row)!=set(FIELDS):raise ValueError('Complete native physical fields required')
        for name in FIELDS:
            value=row[name];dtype=torch.float64 if name=='pressure' else torch.float32
            if value.device.type!=device or value.dtype!=dtype or not value.is_contiguous():
                raise ValueError('Contiguous native dtype and exact transfer device required')
            width=value.element_size();aligned=((offset+width-1)//width)*width
            if aligned!=offset:gaps.append((offset,aligned-offset))
            size=value.numel()*width;entries.append((step,name,aligned,size,value.shape,dtype));offset=aligned+size
    return entries,offset,gaps

def upload_state(fields):
    entries,size,gaps=layout([fields],'cpu')
    pinned=torch.empty(size,dtype=torch.uint8,pin_memory=True)
    for offset,length in gaps:pinned.narrow(0,offset,length).zero_()
    for _,name,offset,length,shape,dtype in entries:
        pinned.narrow(0,offset,length).copy_(fields[name].reshape(-1).view(torch.uint8))
    packed=torch.empty(size,dtype=torch.uint8,device='cuda')
    packed.copy_(pinned,non_blocking=True)
    result=UploadedState({name:packed.narrow(0,offset,length).view(dtype).reshape(shape)
                          for _,name,offset,length,shape,dtype in entries})
    result._pinned_input=pinned;result._device_input=packed
    return result

def download_outputs(outputs):
    entries,size,gaps=layout(outputs,'cuda')
    pinned=torch.empty(size,dtype=torch.uint8,pin_memory=True)
    for offset,length in gaps:pinned.narrow(0,offset,length).zero_()
    result=[{} for _ in outputs]
    for step,name,offset,length,shape,dtype in entries:
        target=pinned.narrow(0,offset,length).view(dtype).reshape(shape)
        target.copy_(outputs[step][name],non_blocking=True)
        result[step][name]=target
    # Every output owns a non-overlapping view into this new allocation. Future
    # calls allocate new storage, so retained outputs cannot be overwritten.
    torch.cuda.current_stream().synchronize()
    return result
