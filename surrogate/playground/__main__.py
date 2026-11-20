"""Start the playground: python -m surrogate.playground --backend composition --grid 32 --port 8095

The GPU backends need a PyTorch with CUDA. When this interpreter has none, the playground restarts itself under
--python, else $PLAYGROUND_PYTHON, else Pumas Library's PyTorch 2.14 runtime when it is installed at its usual place.
"""
import argparse
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PUMAS_PYTHON = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/ai-systems/Pumas-Library/torch-versions/'
                    'v2.14.0/venv/bin/python')
GPU_BACKENDS = ('composition', 'generator')


def grid(text):
    parts = text.lower().split('x')
    try:
        counts = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError('A grid is N (a cube) or NXxNYxNZ, e.g. 32 or 16x24x40.') from None
    if len(counts) == 1:
        counts *= 3
    if len(counts) != 3:
        raise argparse.ArgumentTypeError('A grid is N (a cube) or NXxNYxNZ, e.g. 32 or 16x24x40.')
    return tuple(counts)


def parser():
    p = argparse.ArgumentParser(prog='python -m surrogate.playground', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', default='composition', choices=('composition', 'generator', 'reference', 'fake'))
    p.add_argument('--grid', type=grid, default=(32, 32, 32), help='N or NXxNYxNZ (default 32)')
    p.add_argument('--cells-per-metre', type=float, default=16.0)
    p.add_argument('--tracer', default='layer', choices=('layer', 'stripes', 'ball', 'none'))
    p.add_argument('--mode', default='realtime', choices=('realtime', 'fastest'))
    p.add_argument('--port', type=int, default=8095)
    p.add_argument('--python', help='interpreter with CUDA PyTorch for the GPU backends')
    p.add_argument('--verbose', action='store_true', help='log every request')
    return p


def cuda_python(args):
    """The interpreter to restart under, or None to stay in this one."""
    if args.backend not in GPU_BACKENDS or os.environ.get('PLAYGROUND_REEXEC') == '1':
        return None
    import torch
    if torch.cuda.is_available():
        return None
    chosen = args.python or os.environ.get('PLAYGROUND_PYTHON') or (str(PUMAS_PYTHON) if PUMAS_PYTHON.is_file() else None)
    if not chosen:
        sys.exit(f'The {args.backend} backend needs CUDA PyTorch; pass --python or set PLAYGROUND_PYTHON '
                 '(or start with --backend fake).')
    return chosen


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    python = cuda_python(args)
    if python:
        print(f'Restarting under {python} for CUDA.', flush=True)
        env = dict(os.environ, PLAYGROUND_REEXEC='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=os.pathsep.join(filter(None, [str(REPOSITORY), os.environ.get('PYTHONPATH')])))
        os.chdir(REPOSITORY)
        os.execve(python, [python, '-m', 'surrogate.playground', *argv], env)

    from .backends import SceneSpec, backend_class
    from .loop import Simulation
    from .server import HOST, Playground, serve

    cls = backend_class(args.backend)
    ok, reason = cls.available()
    if not ok:
        sys.exit(f'{cls.label} cannot run here: {reason}')
    spec = SceneSpec(tuple(args.grid), args.cells_per_metre, args.tracer)
    backend = cls()
    simulation = Simulation(backend, spec, mode=args.mode)
    print(f'Preparing {cls.label} on {"x".join(map(str, spec.counts))} at {spec.cells_per_metre:g} cells/m…',
          flush=True)
    try:
        backend.check(spec)
        simulation.start()
    except ValueError as error:
        simulation.stop()
        sys.exit(str(error))
    server = serve(Playground(simulation), args.port, quiet=not args.verbose)
    print(f'Playground at http://{HOST}:{server.server_address[1]}/  (Ctrl+C stops it)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        simulation.stop()


if __name__ == '__main__':
    main()
