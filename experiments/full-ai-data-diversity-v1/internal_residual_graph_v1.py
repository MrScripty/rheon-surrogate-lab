"""CUDAgraph execution of the unchanged managed-restored residual generator."""
import torch

from internal_residual_generator_v1 import InternalResidualGenerator
from cuda_native_transfer_v1 import upload_state, download_outputs


class InternalResidualGraphGenerator(InternalResidualGenerator):
    def __init__(self, *args):
        super().__init__(*args)
        self.graphs = {}
        self.geometry = None
        self.asset_identity.update(execution='CUDAgraph_internal_residual_v1',
            graph_cache='At most four K graphs for one domain shape and density',
            learned_parameters_and_arithmetic_unchanged=True)

    def _clear(self):
        if self.graphs:
            torch.cuda.synchronize()
            self.graphs.clear()

    def _body(self, fields, scene, packet, horizons):
        backbone = self.backbone
        external = backbone.controls.generate_from_packet(tuple(scene['counts']), packet)
        outputs = []
        for lead in range(horizons):
            offset = lead * 8
            for width in (2, 2, 3, 1):
                phase = backbone.phase(fields, scene, external, offset, width)
                fields = backbone.corrected(phase, self.network) if width > 1 else backbone.emit(phase)
                offset += width
            outputs.append(fields)
        return outputs

    @torch.no_grad()
    def _resident(self, fields, scene, start, horizons, reference_substeps):
        if horizons not in (1, 2, 3, 4) or reference_substeps not in (1, 8):
            raise ValueError('Known complete output count and original reference regime required')
        if reference_substeps == 1:
            return super().predict(fields, scene, start, horizons, reference_substeps)
        geometry = (tuple(scene['counts']), scene['cells_per_metre'])
        if geometry != self.geometry:
            self._clear()
            self.geometry = geometry
        self.backbone.model.prepare_geometry(geometry[0])
        packet = self.backbone.packet(scene, start, horizons, 8).to('cuda')
        if horizons not in self.graphs:
            static_fields = {n: v.clone() for n, v in fields.items()}
            static_packet = packet.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._body(static_fields, scene, static_packet, horizons)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._body(static_fields, scene, static_packet, horizons)
            self.graphs[horizons] = (graph, static_fields, static_packet, outputs)
        graph, static_fields, static_packet, outputs = self.graphs[horizons]
        for n, value in fields.items():
            if value.device.type != 'cuda' or value.shape != static_fields[n].shape or value.dtype != static_fields[n].dtype:
                raise ValueError('Exact native CUDA input types and shapes required')
            static_fields[n].copy_(value)
        static_packet.copy_(packet)
        graph.replay()
        return outputs

    @torch.no_grad()
    def predict(self, fields, scene, start, horizons, reference_substeps):
        # Public resident outputs own storage independently of future replays.
        return [{n: v.clone() for n, v in row.items()} for row in
            self._resident(fields, scene, start, horizons, reference_substeps)]

    @torch.no_grad()
    def predict_native(self, fields, scene, start, horizons, reference_substeps):
        return download_outputs(self._resident(upload_state(fields), scene, start, horizons, reference_substeps))

    def close(self):
        self._clear()
        super().close()
