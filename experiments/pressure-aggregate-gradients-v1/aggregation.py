"""Local gradient geometry from streamed float64 dot products; never optimize."""
import math


def summarize(full, cross, term_norms, *, cap=None, global_cap=1., epsilon=1e-6):
    count = len(full)
    if not count or global_cap <= 0 or epsilon < 0 or (cap is not None and cap <= 0):
        raise ValueError('Nonempty geometry and positive finite norm caps required')
    numbers = [global_cap, epsilon] + ([] if cap is None else [cap])
    numbers += [value for row in full for value in row]
    numbers += [value for matrix in cross.values() for row in matrix for value in row]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError('Finite gradient geometry required')
    if any(len(row) != count for row in full) or any(full[i][i] < 0 for i in range(count)):
        raise ValueError('Square nonnegative norm geometry required')
    if set(cross) != set(term_norms) or any(len(matrix) != count or any(len(row) != count for row in matrix) for matrix in cross.values()):
        raise ValueError('Matching square component geometry required')
    if any(len(values) != count or any(not math.isfinite(v) or v < 0 for v in values) for values in term_norms.values()):
        raise ValueError('Finite nonnegative component norms required')
    norms = [math.sqrt(full[i][i]) for i in range(count)]
    scales = [1. if cap is None else min(1., cap/(norm+epsilon)) for norm in norms]
    square = math.fsum(scales[i]*scales[j]*full[i][j] for i in range(count) for j in range(count))/count**2
    if square < 0:
        raise ValueError('Nonnegative aggregate norm required')
    norm = math.sqrt(square)
    final_scale = min(1., global_cap/(norm+epsilon)) if norm+epsilon else 1.
    total_norm = math.fsum(norms)
    cases = []
    for i in range(count):
        dot = math.fsum(scales[j]*full[i][j] for j in range(count))/count
        terms = {}
        for name, matrix in cross.items():
            value = math.fsum(scales[j]*matrix[i][j] for j in range(count))/count
            term_norm = term_norms[name][i]
            terms[name] = dict(dot_mean=value,
                cosine_mean=value/term_norm/norm if term_norm and norm else None,
                derivative_along_negative_clipped_mean=-final_scale*value)
        cases.append(dict(case=i, gradient_l2=norms[i],
            norm_budget_share=norms[i]/total_norm if total_norm else None,
            pre_mean_cap_scale=scales[i], dot_mean=dot,
            cosine_mean=dot/norms[i]/norm if norms[i] and norm else None,
            signed_mean_contribution=scales[i]*dot/count/square if square else None,
            derivative_along_negative_clipped_mean=-final_scale*dot, terms=terms))
    return dict(per_case_cap=cap, mean_l2=norm, global_clip_scale=final_scale,
        clipped_mean_l2=norm*final_scale, cases=cases,
        interpretation='For direction -clip(mean), negative directional derivative means local loss descent, not measured pressure improvement or an Adam update.')
