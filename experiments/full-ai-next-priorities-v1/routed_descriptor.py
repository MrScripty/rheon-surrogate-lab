"""Inference-only assembly of two learned heads with the original shared fine CNN."""
import copy


def describe(previous,spec):
    result=copy.deepcopy(previous)
    for key in ('fine_tuning','method_comparison','training_band_feedback','current_policy_training_feedback',
                'direction_training_feedback','analytical_training_feedback','supplemental_training_feedback','intermediate_feedback'):
        result.pop(key,None)
    result['parameter_count']=94956
    result['neural_head_routing']=copy.deepcopy(spec)
    result['initialization']='Local weight assembly of verified managed CPU restores; zero new optimizer updates'
    result['shape_informed_coarse'].update(network='Two learned3->16tanh->63heads; one selected by observable grid counts',
        additional_head_parameters=1198,fine_initialization='Exact unchanged C107 fine CNN',head_learning_rate_multiplier=None)
    return result
