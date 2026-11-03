"""Current generic CPU adapter entrypoint; historical decoding is offline only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dependency-migration-v1'))
from model_adapter import main

if __name__ == '__main__':
    sys.exit(main())
