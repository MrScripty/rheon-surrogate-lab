"""Exercise the exact captured policy before any managed admission."""
import argparse
import json
from pathlib import Path
import sys


def check(project, expect_rejection=False):
    sys.path.insert(0,str(project))
    from policy import experiment
    config=json.loads((project/'config.json').read_bytes())
    if expect_rejection:
        from support import ContractError
        try:
            experiment(config)
        except ContractError as error:
            assert error.code == 'resources'
            assert str(error) == 'resources: bounded elapsed runtime and complete artifact budget'
            print('Exact historical resource diagnostic reproduced',flush=True)
            return
        raise AssertionError('Historical budget rejection not detected')
    result=experiment(config)
    assert result['resources']['max_output_bytes']==64*1024**2
    provider=json.loads((project/'provider.json').read_bytes())
    assert provider['resources']['output_bytes']>=result['resources']['max_output_bytes']
    print('Captured experiment and provider accept the declared 64 MiB budget',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--expect-resources-rejection',action='store_true')
    args=parser.parse_args()
    check(args.project.resolve(),args.expect_resources_rejection)
