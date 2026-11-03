"""Re-evaluate actual errors after changing neural pressure-map arithmetic."""
import argparse
from pathlib import Path
import evaluate_coupled as core
import temporal_displacement_objective as teaching
from float32_map_model_support import load_model
from study_support import ROOT,HERE,DOCS,read,write,sha


def run(args):
    protocol_path=DOCS/'float32-learned-map-protocol-01.json';protocol=read(protocol_path)
    for name,digest in protocol['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered float32 execution source changed')
    for relative,digest in protocol['execution_assets_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Registered execution asset changed')
    if sha(args.bank)!=protocol['bank_sha256']:raise ValueError('Original development population required')
    def forbidden(*args,**kwargs):raise ValueError('Training labels reached inference')
    teaching.displacement_labels=forbidden;core.load_model=load_model;core.run(args)
    write(args.output/'wrapper-receipt.json',dict(status='completed',wrapper_sha256=sha(__file__),
        actual_loader_sha256=sha(HERE/'float32_map_model_support.py'),unchanged_core_sha256=sha(HERE/'evaluate_coupled.py'),
        core_summary_sha256=sha(args.output/'summary.json'),protocol_sha256=sha(protocol_path),
        raw_core_legacy_loader_stamp_preserved=True,training_teacher_disabled=True,optimizer_updates=0,
        original_accuracy_rules_unchanged=True,exact_physical_output_inheritance_claim=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
