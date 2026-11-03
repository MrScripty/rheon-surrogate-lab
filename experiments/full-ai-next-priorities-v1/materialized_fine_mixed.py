"""Float64 learned-map construction with float32 storage and inference."""
from materialized_fine_precise import materialize as construct_double


def materialize(network,counts):
    matrix,receipt=construct_double(network,counts)
    stored=matrix.float().contiguous()
    return stored,dict(receipt,construction_matrix_bytes=receipt['matrix_bytes'],
        matrix_bytes=stored.numel()*stored.element_size(),
        storage_dtype='float32',construction_dtype='float64',
        method='Float64 unit responses of frozen learned CNN weights, then float32 storage; no physical inverse')
