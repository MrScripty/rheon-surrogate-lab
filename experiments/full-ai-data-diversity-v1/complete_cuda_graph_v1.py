"""Complete GPU control/dynamics graph with dynamic known-input packets.

Every field and known time/control packet is supplied each call. Graph setup is
explicit. CPU packet buffers remain owned until their actual H2D event completes;
all returned physical states have independent GPU storage across future calls.
"""
import time
import torch
from fused_cuda_generator_v1 import FusedCudaGenerator
from dynamic_cuda_controls_v1 import CudaKnownControls
from cuda_divisor_generator_v1 import FIELDS

class CompleteCudaGraphGenerator:
    def __init__(self,pair,shared_pair,build,control_build):
        self.model=FusedCudaGenerator(pair,shared_pair,build,'transport_and_linear')
        self.controls=CudaKnownControls(control_build)
        self.asset_identity=dict(self.model.asset_identity,execution='complete dynamic-control CUDA graph',
            known_controls=self.controls.identity,known_input_packet_columns=18,
            graph_includes='All GPU coordinate/phase/force generation and every learned internal advance; all complete K outputs',
            replay_scope='All current-field D2D copies, one compact known-input H2D packet, graph replay, independent complete output clones',
            cpu_packet_lifetime='Retained until actual recorded H2D event completes; maximum64inflight before bounded event wait')
        self.graphs={};self.setup_observations=[];self.pending_packets=[]

    def state_dict(self):return self.model.state_dict()
    def prepare_geometry(self,counts):self.model.prepare_geometry(counts)

    @torch.inference_mode()
    def prepare_block(self,fields,scene,step,horizons,substeps=1):
        key=(tuple(scene['counts']),scene['cells_per_metre'],horizons,substeps)
        if key in self.graphs:return
        torch.cuda.synchronize();tick=time.perf_counter()
        state={name:value.clone() for name,value in fields.items()}
        packet=self.controls.parameters(scene,step,horizons,substeps).to('cuda')
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                external=self.controls.generate_from_packet(key[0],packet)
                self.model.predict(state,scene,step,horizons,external,substeps)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            external=self.controls.generate_from_packet(key[0],packet)
            outputs=self.model.predict(state,scene,step,horizons,external,substeps)
        torch.cuda.synchronize();self.graphs[key]=(graph,state,packet,outputs)
        self.setup_observations.append(dict(counts_xyz=list(key[0]),cells_per_metre=key[1],K=horizons,
            internal_substeps=substeps,internal_advances_per_block=horizons*substeps,graph_setup_ms=1000*(time.perf_counter()-tick)))

    @torch.inference_mode()
    def predict(self,fields,scene,step,horizons,substeps=1):
        key=(tuple(scene['counts']),scene['cells_per_metre'],horizons,substeps)
        if key not in self.graphs or set(fields)!=set(FIELDS) or scene.get('sgs',False):raise ValueError('Prepared complete native CUDA state required')
        graph,state,packet,outputs=self.graphs[key]
        # Different scenes, times and pulse/center programs with this geometry
        # remain dynamic graph inputs; no force field is baked into replay.
        known=self.controls.parameters(scene,step,horizons,substeps).pin_memory()
        if known.shape!=packet.shape:raise ValueError('Complete fixed-shape compact control packet required')
        self.pending_packets=[entry for entry in self.pending_packets if not entry[0].query()]
        if len(self.pending_packets)>=64:
            self.pending_packets[0][0].synchronize()
            self.pending_packets=[entry for entry in self.pending_packets if not entry[0].query()]
        for name,target in state.items():
            source=fields[name]
            if source.device!=target.device or source.dtype!=target.dtype or source.shape!=target.shape or not source.is_contiguous():
                raise ValueError('Exact native current CUDA state layout required')
            target.copy_(source)
        packet.copy_(known,non_blocking=True)
        ready=torch.cuda.Event();ready.record(torch.cuda.current_stream());self.pending_packets.append((ready,known))
        graph.replay()
        return [{name:value.clone() for name,value in row.items()} for row in outputs]

    def release_graphs(self):
        torch.cuda.synchronize();self.graphs.clear();self.pending_packets.clear()

    def close(self):
        self.release_graphs();self.controls.close();self.model.close()
