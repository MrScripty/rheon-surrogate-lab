"""Describe the unchanged pressure graph and the actual head calibration protocol."""
import copy


def describe(previous,calibration):
    result=copy.deepcopy(previous)
    for key in ('fine_tuning','method_comparison','training_band_feedback','current_policy_training_feedback',
                'direction_training_feedback','analytical_training_feedback','supplemental_training_feedback','intermediate_feedback'):
        result.pop(key,None)
    result['training_objective']='offline modal teacher log-gain calibration; fine network frozen'
    result['head_calibration']=copy.deepcopy(calibration)
    result['shape_informed_coarse'].update(head_learning_rate_multiplier=1,
        fine_initialization='Exact C107 fine tensors frozen throughout calibration')
    result['learned_coarse_revision']['initialization']='Exact C107 learned head, followed by offline modal calibration'
    return result
