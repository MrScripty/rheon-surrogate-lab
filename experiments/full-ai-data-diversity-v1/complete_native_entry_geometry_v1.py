"""Complete native entry combined with qualified generic integer geometry."""
from complete_native_entry_v1 import CompleteNativeEntry
from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph


class GeometryCompleteNativeEntry(CompleteNativeEntry):
    generator_class = GeometrySpecializedGraph
