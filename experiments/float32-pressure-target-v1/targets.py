"""Model-output acceptance targets, separate from historical native solver gates."""
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    value = json.loads(Path(path).read_text())
    require(set(value) == {'schema', 'id', 'output_dtype', 'relative_residual', 'absolute_residual', 'calibration'},
            'closed float32 target profile')
    require(value['schema'] == 'rheon-float32-pressure-target-v1' and value['output_dtype'] == 'float32'
            and isinstance(value['id'], str) and value['id'], 'float32 profile identity')
    for name in ('relative_residual', 'absolute_residual'):
        require(type(value[name]) in (int, float) and math.isfinite(value[name]) and value[name] > 0,
                'positive finite '+name)
    require(value['absolute_residual'] == 1e-12, 'preserve original absolute residual floor')
    calibration = value['calibration']
    require(set(calibration) == {'population', 'safety_factor', 'minimum_relative_residual'}
            and calibration['population'] == 'train_only', 'train-only calibration')
    for name in ('safety_factor', 'minimum_relative_residual'):
        require(type(calibration[name]) in (int, float) and math.isfinite(calibration[name])
                and calibration[name] > 0, 'positive finite '+name)
    require(calibration['safety_factor'] >= 1, 'calibration margin must not shrink measured error')
    return value


def calibrated_relative_target(training_peak, calibration):
    require(type(training_peak) in (int, float) and math.isfinite(training_peak) and training_peak >= 0,
            'finite nonnegative training rounding peak')
    required = max(training_peak*calibration['safety_factor'], calibration['minimum_relative_residual'])
    require(math.isfinite(required) and required > 0, 'finite calibration requirement')
    return 10.0**math.ceil(math.log10(required))


def assess(native_quality, profile):
    """Consume quality already checked by the unchanged native-response validator.

    Change only the original-RHS residual gate. Every native physical failure
    remains a failure, including compact shape/gauge/nonfinite rejection.
    This does not relabel the underlying native solve or its iteration count.
    """
    failures = list(native_quality['failures'])
    if set(native_quality) == {'qualified', 'failures'}:
        require(native_quality['qualified'] is False and failures, 'invalid compact quality')
        return {'profile': profile['id'], 'qualified': False, 'failures': failures, 'residual_limit': None}
    require(type(native_quality['qualified']) is bool and native_quality['qualified'] is (not failures)
            and len(failures) == len(set(failures)), 'validated native outcome required')
    residual, rhs = native_quality['residual_l2'], native_quality['rhs_l2']
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (residual, rhs)),
            'finite nonnegative native norms')
    limit = max(profile['absolute_residual'], profile['relative_residual']*rhs)
    failures = [reason for reason in failures if reason != 'original_rhs_full_residual']
    if residual > limit:
        failures.insert(0, 'original_rhs_full_residual')
    return {'profile': profile['id'], 'qualified': not failures, 'failures': failures,
            'residual_limit': limit, 'residual_to_target_ratio': residual/limit}
