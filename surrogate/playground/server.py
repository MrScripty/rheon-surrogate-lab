"""The playground's HTTP server: the page, and a small JSON API over the simulation. Binds 127.0.0.1 only.

    GET  /                     the page
    GET  /static/<file>        its scripts and style
    GET  /api/capabilities     backends (and whether each runs here), the current backend's grids, fields, modes
    GET  /api/frame            ?views=z:16:speed,x:16:tracer&arrows=1 -> the planes, stats, forces, status
    POST /api/backend          {backend, counts, cells_per_metre, tracer} switch backend (and reset)
    POST /api/reset            {counts, cells_per_metre, tracer}
    POST /api/forces           {forces: [blob, ...]} replaces every force
    POST /api/tracer           {pattern}
    POST /api/control          {action: pause | resume | step | mode, mode, steps}

Errors answer 400 (or 409 while the grid is being prepared) with {error}.
"""
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import frames
from .backends import SceneSpec, Unsupported, backend_class, listing
from .loop import MODES

STATIC = Path(__file__).resolve().parent / 'static'
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
         '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml'}
MAX_BODY = 64 * 1024
HOST = '127.0.0.1'


class Playground:
    """What the request handlers share: the simulation and a cache of the last frame."""

    def __init__(self, simulation):
        self.simulation = simulation
        self._cache_lock = threading.Lock()
        self._cache = (None, None)

    def capabilities(self):
        backend = self.simulation.backend
        return dict(backends=listing(), current=backend.capabilities() if backend else None,
                    info=backend.info() if backend else None, modes=list(MODES))

    def frame(self, query):
        sim = self.simulation
        snapshot = sim.snapshot()
        backend = sim.backend
        result = dict(snapshot, info=backend.info() if backend else None,
                      forces=[f.to_json() for f in backend.forces] if backend else [], views=[])
        if snapshot['status'] in ('loading',) or backend is None or backend.spec is None:
            return result
        counts = backend.spec.counts
        requested = []
        for text in (query.get('views', ['z::speed'])[0] or 'z::speed').split(',')[:2]:
            parts = (text.split(':') + ['', '', ''])[:3]
            requested.append(frames.parse_view(dict(axis=parts[0] or 'z', index=parts[1], field=parts[2] or 'speed'),
                                               counts))
        with_arrows = query.get('arrows', ['0'])[0] == '1'
        key = (snapshot['version'], id(backend), json.dumps(requested), with_arrows)
        with self._cache_lock:
            if self._cache[0] == key:
                result['views'] = self._cache[1]
                return result
        try:
            # Version and planes are read together on the simulation thread, between steps.
            version, planes = sim.call(lambda: (sim.version, frames.render(sim.backend, requested, with_arrows)),
                                       timeout=5)
        except TimeoutError:
            return result
        result.update(sim.snapshot(), views=planes)
        with self._cache_lock:
            self._cache = ((version, id(backend), key[2], with_arrows), planes)
        return result

    def spec(self, body):
        return SceneSpec.from_json(body, default=self.simulation.spec)

    def reset(self, body):
        spec = self.spec(body)
        sim = self.simulation
        return dict(info=sim.call(lambda: sim._reset(spec), timeout=600), capabilities=self.capabilities())

    def switch(self, body):
        name = body.get('backend')
        cls = backend_class(name)
        ok, reason = cls.available()
        if not ok:
            raise Unsupported(f'{cls.label} cannot run here: {reason}')
        spec = self.spec(body)
        sim = self.simulation
        backend = cls()
        backend.check(spec)
        return dict(info=sim.call(lambda: sim._reset(spec, backend=backend), timeout=600),
                    capabilities=self.capabilities())

    def forces(self, body):
        forces = body.get('forces')
        if not isinstance(forces, list):
            raise ValueError('Send {"forces": [...]}, an empty list to remove them all.')
        sim = self.simulation

        def apply():
            sim.backend.set_forces(forces)
            return [f.to_json() for f in sim.backend.forces]
        return dict(forces=sim.call(apply))

    def tracer(self, body):
        sim = self.simulation

        def apply():
            sim.backend.reset_tracer(body.get('pattern', 'layer'))
            sim.version += 1
            return sim.snapshot()
        return sim.call(apply)

    def control(self, body):
        return self.simulation.control(body.get('action'), mode=body.get('mode'), steps=body.get('steps', 1))


def handler_for(playground, quiet=True):
    routes = {'/api/backend': playground.switch, '/api/reset': playground.reset, '/api/forces': playground.forces,
              '/api/tracer': playground.tracer, '/api/control': playground.control}

    class Handler(BaseHTTPRequestHandler):
        server_version = 'playground'

        def log_message(self, format, *args):
            if not quiet:
                super().log_message(format, *args)

        def _send(self, status, body, kind='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def _error(self, error):
            status = HTTPStatus.CONFLICT if playground.simulation.status == 'loading' else HTTPStatus.BAD_REQUEST
            self._send(status, dict(error=str(error)))

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path in ('/', '/index.html'):
                    return self._static('index.html')
                if url.path.startswith('/static/'):
                    return self._static(url.path[len('/static/'):])
                if url.path == '/api/capabilities':
                    return self._send(200, playground.capabilities())
                if url.path == '/api/frame':
                    return self._send(200, playground.frame(parse_qs(url.query)))
            except (ValueError, KeyError, RuntimeError) as error:
                return self._error(error)
            self._send(404, dict(error='Not found.'))

        def _static(self, name):
            files = {p.name: p for p in STATIC.iterdir() if p.is_file()}
            path = files.get(name)
            if path is None or path.suffix not in TYPES:
                return self._send(404, dict(error='Not found.'))
            self._send(200, path.read_bytes(), TYPES[path.suffix])

        def do_POST(self):
            route = routes.get(urlparse(self.path).path)
            if route is None:
                return self._send(404, dict(error='Not found.'))
            try:
                length = int(self.headers.get('Content-Length') or 0)
                if not 0 <= length <= MAX_BODY:
                    raise ValueError('The request body is too large.')
                body = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(body, dict):
                    raise ValueError('The request body is a JSON object.')
                self._send(200, route(body))
            except json.JSONDecodeError:
                self._error(ValueError('The request body is not JSON.'))
            except (ValueError, KeyError, RuntimeError, TimeoutError) as error:
                self._error(error)

    return Handler


def serve(playground, port, quiet=True):
    """A server bound to 127.0.0.1:`port` (0 picks a free port). Call serve_forever() on it."""
    server = ThreadingHTTPServer((HOST, port), handler_for(playground, quiet))
    server.daemon_threads = True
    return server
