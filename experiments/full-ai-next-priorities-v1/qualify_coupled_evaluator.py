"""Replay a verified model and injected numerical failure through the new evaluator."""
import argparse
from pathlib import Path
import evaluate_coupled as evaluation
from decoder_evaluation_support import load_model
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh evaluator qualification required')
    args.output.mkdir(parents=True)
    protocol=DOCS/'temporal-coupled-evaluation-protocol-01.json'
    if read(protocol)['evaluator_sha256']!=sha(HERE/'evaluate_coupled.py'):raise ValueError('Registered evaluator changed')
    evaluation.load_model=load_model
    pair=BASE/'b2-endpoint-decoder-pair-01';bank=BASE/'temporal-development-240-01/bank.json'
    class Arguments:pass
    finite=Arguments();finite.pair=pair;finite.bank=bank;finite.output=args.output/'finite'
    evaluation.run(finite)
    reference=BASE/'temporal-decoder-evaluation-01/b2-endpoint-decoder';expected=read(reference/'summary.json');actual=read(finite.output/'summary.json')
    if len(actual['scenes'])!=8 or actual['bank_sha256']!=expected['bank_sha256']or actual['metrics_sha256']!=expected['metrics_sha256']:
        raise ValueError('Finite-case evaluation population or metrics differ')
    outputs=0
    for scene in actual['scenes']:
        name=scene['identifier']+'.json';old=read(reference/name);new=read(finite.output/name)
        if old['rows']!=new['rows']or old['summaries']!=new['summaries']or new['inference_failures']:
            raise ValueError('Finite candidate metrics changed')
        outputs+=len(new['rows'])
    one=read(bank);one['trajectories']=one['trajectories'][:1];write(args.output/'one-scene.json',one)
    def injected_loader(path):
        network,norm,identity=load_model(path)
        class Injected:
            horizons=network.horizons
            last=None
            own=False
            def predict(self,fields,scene,step,normalization):
                if step==0:self.own=self.last==120
                if self.own and step==12:raise FloatingPointError('Qualified injected raw numerical failure')
                self.last=step
                return network.predict(fields,scene,step,normalization)
        return Injected(),norm,dict(identity,evaluator_qualification_injection=True,candidate_frontier_point=False)
    evaluation.load_model=injected_loader
    failure=Arguments();failure.pair=pair;failure.bank=args.output/'one-scene.json';failure.output=args.output/'injected'
    evaluation.run(failure)
    observed=read(failure.output/'summary.json')['scenes'][0]
    if observed['first_unavailable_step']!=13 or len(observed['inference_failures'])!=1:
        raise ValueError('Raw failed block availability not recorded')
    if not observed['summaries'][1]['complete']or observed['summaries'][2]['complete']or observed['summaries'][3]['complete']:
        raise ValueError('Failed-block interval completeness incorrect')
    if observed['summaries'][3]['all_accuracy_criteria_pass']:
        raise ValueError('Incomplete rollout cannot pass accuracy')
    write(args.output/'receipt.json',dict(status='passed',all_eight_finite_scenes_metrics_bitwise_equal=True,
        finite_outputs_compared=outputs,injected_failed_block_first_unavailable_step=13,
        no_fabricated_outputs_or_fallback=True,original_error_criteria_unchanged=True,optimizer_updates=0,
        candidate_frontier_point=False,evaluator_sha256=sha(HERE/'evaluate_coupled.py'),
        source_sha256=sha(__file__),protocol_sha256=sha(protocol),
        reference_summary_sha256=sha(reference/'summary.json'),finite_summary_sha256=sha(finite.output/'summary.json'),
        injected_summary_sha256=sha(failure.output/'summary.json')))
    print('Qualified unchanged finite metrics and explicit raw failed-block accounting',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
