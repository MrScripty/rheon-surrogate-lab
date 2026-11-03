"""Offline exact-displacement control; never a full-AI candidate or speed result."""
import argparse
from pathlib import Path
from profile_frozen_components import read,write
import diagnose_physical_transport as diagnosis
import physical_transport_composition as composition
from training_displacement_teacher import displacement_labels


class OfflineOracleComposition(composition.PhysicalTransportComposition):
    def __init__(self,parent,mode):
        if mode!='offline_exact_displacement':raise ValueError('Registered offline oracle only')
        super().__init__(parent,'physical_carriers_and_point')


def exact_teacher_shift(head,velocity,normalized,hidden,alpha,dt_over_h):
    counts=tuple(int(size)-1 for size in reversed(velocity.shape[2:]))
    shifts,unused_first_velocity=displacement_labels(velocity,counts,dt_over_h)
    return shifts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    original_model=diagnosis.PhysicalTransportComposition
    original_shift=composition.physical_point_shift
    diagnosis.PhysicalTransportComposition=OfflineOracleComposition
    composition.physical_point_shift=exact_teacher_shift
    try:diagnosis.run(args)
    finally:
        diagnosis.PhysicalTransportComposition=original_model
        composition.physical_point_shift=original_shift
    result=read(args.output/'results.json')
    result.update(status='completed_offline_exact_displacement_control',native_teacher_displacement_used=True,
        full_ai_candidate=False,causal_full_ai_generator=False,candidate_frontier_point=False,latency_claim=False,
        purpose='Attribute frozen neural displacement error versus remaining prepared-RHS components; no teacher component may enter a scored full-AI candidate')
    write(args.output/'results.json',result)
