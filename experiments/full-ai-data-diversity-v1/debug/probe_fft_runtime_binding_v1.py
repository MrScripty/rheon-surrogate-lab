"""Minimal runtime-constructor regression: no FFT or model forward is run."""
import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh runtime observation required')
    result = dict(status='running', module=args.module, model_forward=False, FFT_executed=False)
    try:
        module = importlib.import_module(args.module)
        owner = module.FFTDCT()
        result.update(status='passed_actual_FFT_constructor_binding', identity=owner.identity)
    except BaseException as e:
        result.update(status='failed_actual_FFT_constructor_binding', error=repr(e))
        raise
    finally:
        paths = {Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
                 if 'libcufft.so.' in line and line.split()[-1].startswith('/')}
        result['loaded_cufft'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(paths)]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
