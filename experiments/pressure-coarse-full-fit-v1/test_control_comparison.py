"""Actual retained campaign/package fixtures reject wrong comparator identity."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from control_comparison import bind_campaign


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','config','weights','different-weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();campaign=json.loads(args.campaign.read_text())
    owner=Path(__file__).resolve().parents[1]/'pressure-throughput-feedback-v1/benchmark.py'
    actual=digest(args.weights);other=digest(args.different_weights);assert actual!=other
    bind_campaign(campaign,args.config,actual,owner)
    wrong=copy.deepcopy(campaign);wrong['manifest_sha256']='0'*64
    results=[]
    for label,value,expected_weight,message in (
        ('wrong-campaign-package',wrong,actual,'Control campaign package manifest differs'),
        ('wrong-control-weights',campaign,other,'Control campaign packaged weights differ from measured Study')):
        try:bind_campaign(value,args.config,expected_weight,owner)
        except ValueError as error:
            assert str(error)==message
            results.append(dict(case=label,expected=message,observed=str(error),passed=True))
        else:raise AssertionError('Wrong comparator accepted')
    result=dict(status='passed',actual_retained_campaign_accepted=True,negative_cases=results,
        scope='Real retained campaign/config/package; one altered manifest identity, then a different valid checkpoint identity supplied as expected control. No model execution or campaign regeneration.',
        sources={str(p):digest(p)for p in(args.campaign,args.config,args.weights,args.different_weights,Path(__file__),Path(__file__).with_name('control_comparison.py'))})
    with args.output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='passed',negative_cases=len(results))))


if __name__=='__main__':main()
