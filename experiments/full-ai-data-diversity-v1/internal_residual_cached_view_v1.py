"""Own exact training tensors on CUDA so repeated epochs need no archive reads."""
import torch


class GPUCachedTrainingView:
    def __init__(self, source, maximum_allocated_bytes, maximum_reserved_bytes, progress=None):
        self.source = source
        self.active_weight = source.active_weight
        self.sample_gradient_weight = source.sample_gradient_weight
        self.records = []
        self.cached_tensor_bytes = 0
        pointers = set()
        for index in range(783):
            row, sample, response = source.sample(index)
            for value in sample.values():
                if not isinstance(value, torch.Tensor):
                    continue
                if value.device.type != 'cuda' or value.requires_grad or value.dtype not in (torch.float32, torch.float64):
                    raise ValueError('Immutable CUDA training-input tensors required')
                if value.data_ptr() in pointers:
                    raise ValueError('Different cached training fields share caller storage')
                pointers.add(value.data_ptr())
                self.cached_tensor_bytes += value.numel() * value.element_size()
            self.records.append((row, sample, response))
            if (torch.cuda.max_memory_allocated() > maximum_allocated_bytes or
                    torch.cuda.max_memory_reserved() > maximum_reserved_bytes):
                raise ValueError('Declared training-cache CUDA budget exceeded')
            if progress is not None:
                progress(index + 1)

    def sample(self, index):
        return self.records[index]

    def fixed_objective(self, normalization):
        return self.source.fixed_objective(normalization)

    def close(self):
        self.records.clear()
