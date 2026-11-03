"""CPU scoring of already generated component fields;never runs a model."""
import numpy as np


def masked_errors(prediction,target,mask):
    prediction,target,mask = [np.asarray(value,dtype='<f8') for value in (prediction,target,mask)]
    if prediction.shape != target.shape or prediction.shape != mask.shape \
            or not all(np.isfinite(value).all() for value in (prediction,target,mask)) \
            or not np.isin(mask,[0.,1.]).all() or not np.count_nonzero(mask):
        raise ValueError('Finite equal-shaped component fields and nonempty binary face mask required')
    difference = (prediction-target)*mask
    reference = target*mask
    error_norm = float(np.linalg.norm(difference.ravel()))
    target_norm = float(np.linalg.norm(reference.ravel()))
    return dict(masked_relative_l2=error_norm/max(target_norm,1e-30),
        masked_absolute_rms=float(error_norm/np.sqrt(np.count_nonzero(mask))),
        masked_absolute_max=float(np.max(np.abs(difference))),target_masked_l2=target_norm,
        target_denominator_floor_active=target_norm < 1e-30,free_face_values=int(np.count_nonzero(mask)))
