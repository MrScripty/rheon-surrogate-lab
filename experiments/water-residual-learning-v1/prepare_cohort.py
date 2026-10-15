"""Prepare the next varied cohort without starting training or consuming a test."""
import argparse
import json
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(LAB/'experiments/water-learning-sequence-v1'))
    import views
    import data
    selections=[('baseline','train',i,[0,1]) for i in range(16)]+[('baseline','validation',i,[0,1]) for i in range(64)]
    receipt=views.create({'baseline':args.packet.absolute()},selections,'train_validation',args.output.absolute())
    _reservation,parents=data.read_view(args.output.absolute())
    training=[p for p in parents if p['entry']['split']=='train']
    setups=[{'trajectory_id':p['entry']['trajectory_id'],'counts_xyz':p['counts_xyz'],'cell_size_m':p['cell_size_m'],
        'force':p['scene']['force'],'native_frames':[f['index'] for f in p['frames']]} for p in training]
    report={'schema':'rheon-residual-prepared-cohort-v1','status':'prepared_not_trained','training_simulations':16,
        'training_states':32,'validation_simulations':64,'validation_states':128,'new_reserved_test_used':False,
        'qualification_before_training':'known-case physical accuracy gate; no automatic training admission',
        'receipt':receipt,'training_setups':setups}
    (args.output/'cohort-method.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
