"""Evaluate the composed neural model through the unchanged every-output core."""
import argparse
from pathlib import Path
import evaluate_coupled as core
import temporal_displacement_objective as teaching
from fixed_viscosity_evaluation_support import load_model
from study_support import ROOT,HERE,DOCS,read,write,sha


def run(args):
    reg_path=DOCS/'fixed-viscosity-replication-fit-registration-01.json';reg=read(reg_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered full-model evaluation changed')
    if sha(args.bank)!=reg['bank_sha256']:raise ValueError('Registered development scenes changed')
    def forbidden(*args,**kwargs):raise ValueError('Training label function reached inference')
    teaching.displacement_labels=forbidden;core.load_model=load_model
    core.run(args)
    write(args.output/'wrapper-receipt.json',dict(status='completed',wrapper_sha256=sha(__file__),
        actual_loader_sha256=sha(HERE/'fixed_viscosity_evaluation_support.py'),unchanged_core_sha256=sha(HERE/'evaluate_coupled.py'),
        core_summary_sha256=sha(args.output/'summary.json'),registration_sha256=sha(reg_path),
        raw_core_legacy_loader_stamp_preserved=True,training_teacher_disabled=True,optimizer_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
