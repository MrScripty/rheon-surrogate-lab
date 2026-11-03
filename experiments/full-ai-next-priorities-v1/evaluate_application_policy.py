"""Evaluate a concrete deployment policy on the unchanged every-output core.

Its model specification and source protocol must be frozen before invocation.
This is actual accuracy evaluation, including changed rounding; no accuracy is
inherited from a different checkpoint or from the short interface qualification.
"""
import argparse
from pathlib import Path
import evaluate_coupled as core
import temporal_displacement_objective as teaching
from application_model_support import load_model
from study_support import ROOT,HERE,read,write,sha


def run(args):
    protocol=read(args.protocol)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered deployment-policy evaluation source changed')
    if sha(args.specification)!=protocol['model_specification_sha256']or sha(args.bank)!=protocol['bank_sha256']:
        raise ValueError('Exact frozen model specification and reference population required')
    spec=read(args.specification);directory=(ROOT/spec['directory']).resolve()
    def forbidden(*args,**kwargs):raise ValueError('Training labels reached deployment-policy inference')
    teaching.displacement_labels=forbidden
    def load(path):
        if path.resolve()!=directory:raise ValueError('Unexpected deployment artifact path')
        return load_model(spec)
    core.load_model=load
    namespace=argparse.Namespace(pair=directory,bank=args.bank,output=args.output)
    core.run(namespace)
    write(args.output/'wrapper-receipt.json',dict(status='completed',wrapper_sha256=sha(__file__),
        actual_application_loader_sha256=sha(HERE/'application_model_support.py'),
        unchanged_core_sha256=sha(HERE/'evaluate_coupled.py'),core_summary_sha256=sha(args.output/'summary.json'),
        protocol_sha256=sha(args.protocol),model_specification_sha256=sha(args.specification),
        raw_core_legacy_loader_stamp_preserved=True,training_teacher_disabled=True,optimizer_updates=0,
        accuracy_recomputed_for_actual_deployment=True,physical_output_equivalence_inherited=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('specification','protocol','bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
