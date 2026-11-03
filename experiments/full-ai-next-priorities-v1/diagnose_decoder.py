"""Use the original fixed-window fitting diagnostic for a managed global model."""
import argparse
from pathlib import Path
import diagnose_temporal_accuracy as diagnosis
from decoder_evaluation_support import load_model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    diagnosis.load_model=load_model
    diagnosis.run(parser.parse_args())
