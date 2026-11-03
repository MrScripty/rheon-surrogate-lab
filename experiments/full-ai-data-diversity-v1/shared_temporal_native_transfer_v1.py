"""Owned complete CPU-field to CPU-field transaction for shared generation."""
import torch

from cuda_native_transfer_v1 import upload_state, download_outputs


@torch.inference_mode()
def transaction(model, fields, scene, start, rate, k):
    uploaded = outputs = None
    drained = False
    try:
        uploaded = upload_state(fields)
        outputs = model.predict(uploaded, scene, start, rate, k)
        result = download_outputs(outputs)
        drained = True
        return result
    finally:
        if not drained:
            torch.cuda.current_stream().synchronize()
