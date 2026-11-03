"""Separate learned-displacement error from force/viscosity accumulation precision."""
import argparse
from pathlib import Path
from profile_frozen_components import read,write
import diagnose_physical_transport as diagnosis
import double_preparation_composition as composition
from training_displacement_teacher import displacement_labels


def oracle_shift(head,velocity,normalized,hidden,alpha,dt_over_h):
    counts=tuple(int(size)-1 for size in reversed(velocity.shape[2:]))
    return displacement_labels(velocity,counts,dt_over_h)[0]


class RegisteredComposition(composition.DoublePreparationComposition):
    def __init__(self,parent,mode):
        if mode not in ('double_preparation','offline_exact_displacement_double_preparation'):
            raise ValueError('Two registered precision attribution modes only')
        super().__init__(parent,'physical_carriers_and_point');self.oracle=mode.startswith('offline_')

    def predict(self,*args,**kwargs):
        original=composition.physical_point_shift
        if self.oracle:composition.physical_point_shift=oracle_shift
        try:return super().predict(*args,**kwargs)
        finally:composition.physical_point_shift=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    original=diagnosis.PhysicalTransportComposition;diagnosis.PhysicalTransportComposition=RegisteredComposition
    try:diagnosis.run(args)
    finally:diagnosis.PhysicalTransportComposition=original
    result=read(args.output/'results.json')
    result.update(status='completed_double_preparation_precision_attribution',oracle_mode_is_not_full_ai_candidate=True,
        known_forcing_and_learned_viscosity_accumulate_in_double=True,ordinary_mode_uses_only_learned_displacement=True,
        candidate_frontier_point=False,latency_claim=False)
    write(args.output/'results.json',result)
