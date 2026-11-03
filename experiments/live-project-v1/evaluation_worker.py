"""Single-threaded scientific caller; live observation runs only in its parent."""
import argparse
from pathlib import Path
import signal
import sys

PROFILE = Path(__file__).resolve().parents[2]/'experiments/local-pressure-v1'
sys.path.insert(0,str(PROFILE))
import contract
import evaluate


def interrupted(_signal, _frame):
    # Allow the frozen native owner to drain its own session on interruption.
    raise KeyboardInterrupt()


def main():
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    selected=contract.decode(contract.read(args.workspace/'material.json',contract.MIB))
    evaluate.run(Path(selected['capture']),Path(selected['packet']),Path(selected['native_binary']),
                 args.workspace/'execution/stage-00/artifacts',args.workspace/'execution/stage-01/artifacts',args.output)


if __name__=='__main__':main()
