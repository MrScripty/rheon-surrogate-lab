"""Run the unchanged scientific benchmark with current rejection classification."""
from pathlib import Path
import signal
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pressure-throughput-feedback-v1'))
import benchmark
import failure
import rejection


def interrupted(*_):raise KeyboardInterrupt('CPU campaign owner cancelled')


def main():
    # Only the episode exception classifier changes. Every successful arithmetic,
    # stopping policy, oracle and physical result remains owned by the same code.
    failure.classify=rejection.classify
    signal.signal(signal.SIGTERM,interrupted)
    if len(sys.argv)>1 and sys.argv[1]=='episode':
        import episode
        sys.argv[:]=[episode.__file__,*sys.argv[2:]];episode.main()
    else:
        # Preserve this explicit entry in child argv and source receipt identity.
        benchmark.__file__=str(Path(__file__).resolve());benchmark.main()


if __name__=='__main__':main()
