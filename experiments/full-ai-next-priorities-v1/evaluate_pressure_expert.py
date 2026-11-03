"""Reuse the qualified all-horizon and240-step evaluator with a verified global model."""
import argparse
from pathlib import Path
import evaluate_temporal as evaluation
from pressure_expert_evaluation_support import load_model
from study_support import HERE,write,sha


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','bank','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    evaluation.load_model=load_model
    evaluation.run(args)
    write(args.output/'execution-adapter.json',dict(source_sha256=sha(__file__),
        evaluator_sha256=sha(HERE/'evaluate_temporal.py'),loader_sha256=sha(HERE/'pressure_expert_evaluation_support.py'),
        unchanged_metrics_and_populations=True,optimizer_updates=0))
