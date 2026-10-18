"""Small CPU float64 summaries of frozen float32 diagnostic gradients."""
import numpy as np


def projection(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Finite matching one-dimensional vectors required')
    dot = float(left @ right)
    a, b = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    return dict(dot=dot, left_l2=a, right_l2=b, cosine=dot/a/b if a and b else None)


def layers(left, right, layout):
    result, cursor = [], 0
    for entry in layout:
        end = cursor + entry['count']
        result.append(dict(name=entry['name'], count=entry['count'],
                           **projection(left[cursor:end], right[cursor:end])))
        cursor = end
    if cursor != len(left) or cursor != len(right):
        raise ValueError('Complete captured parameter layout required')
    return result
