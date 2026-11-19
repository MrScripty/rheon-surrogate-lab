"""Describe a frozen managed-parent pressure network with a newly trained tail head."""
import copy


def describe(previous,calibration):
    result=copy.deepcopy(previous)
    result['parameter_count']=128938
    result['parent_parameter_count']=94956
    result['initialization']='Verified selected routed C107 successor plus seeded tail head; all parent tensors frozen'
    result['tail_calibration']=copy.deepcopy(calibration)
    result['final_learned_global_block']=dict(modes_per_axis=8,non_DC_gains=511,network='3->64tanh->511 plus511learned logits',
        trainable_parameters=33982,frozen_parent_parameters=94956,small_grid_fine_calls=2,small_grid_global_blocks=3,
        larger_grid_behavior='Unchanged parent prediction; no tail above16cellsperaxis',
        numerical_inverse_at_inference=False,teacher='Offline numerical modal labels, whole-geometry split')
    return result
