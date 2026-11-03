"""Complete native entry using the existing qualified pinned byte transfers."""
import torch

from cuda_native_transfer_v1 import upload_state, download_outputs


@torch.inference_mode()
def transaction(model, fields, scene, start, rate, k, groups):
    uploaded = outputs = None
    drained = False
    try:
        # The owned upload retains both pinned source and GPU byte storage until
        # all queued work is drained, including an exception during prediction.
        uploaded = upload_state(fields)
        outputs = model.predict(uploaded, scene, start, rate, k, groups)
        result = download_outputs(outputs)
        drained = True
        return result
    finally:
        if not drained:
            torch.cuda.current_stream().synchronize()
