"""Use the qualified every-horizon/own240 evaluator with the new restored loader."""
import argparse
from pathlib import Path
import evaluate_coupled as evaluation
import temporal_displacement_objective as teaching
from displacement_evaluation_support import load_model
from study_support import ROOT,HERE,DOCS,read,sha


def run(args):
    protocol=read(DOCS/'temporal-displacement-evaluation-protocol-01.json')
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered displacement evaluation source changed')
    if sha(args.bank)!=protocol['bank_sha256']:raise ValueError('Registered development bank changed')
    def forbidden(*args,**kwargs):raise ValueError('Training teacher reached during forecast evaluation')
    teaching.displacement_labels=forbidden
    evaluation.load_model=load_model
    evaluation.run(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
