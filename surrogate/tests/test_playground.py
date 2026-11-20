"""The playground on the CPU: forces, the fake backend, planes, pacing and the server."""
import base64
import json
import math
import threading
import unittest
import urllib.error
import urllib.request

import torch

from surrogate import forcing
from surrogate.playground import blobs, frames, views
from surrogate.playground.backends import SceneSpec, Unsupported, backend_class, listing
from surrogate.playground.backends.base import DT, tracer_pattern
from surrogate.playground.loop import Pacer, Simulation, Stats
from surrogate.playground.server import Playground, serve


def push(centre=(0.5, 0.3, 0.5), direction=(0, 1, 0), magnitude=20.0, radius=0.15, **extra):
    return dict(centre=list(centre), radius=radius, direction=list(direction), magnitude=magnitude, **extra)


class Blobs(unittest.TestCase):
    def test_a_blob_is_checked_and_normalised(self):
        blob = blobs.Blob.from_json(push(centre=(1.4, -0.2, 0.5), direction=(0, 3, 4)))
        self.assertEqual(blob.centre, (1.0, 0.0, 0.5))
        self.assertAlmostEqual(blob.direction[1], 0.6)
        self.assertEqual(blobs.Blob.from_json(push(direction=(0, 0, 0))).direction, (0.0, 0.0, 0.0))
        for bad in (push(radius=0), push(magnitude=1e9), push(centre=(0, 0)), push(direction=('a', 0, 0)),
                    push(swirl=float('nan')), dict(radius=1), 'blob'):
            with self.assertRaises(ValueError, msg=bad):
                blobs.Blob.from_json(bad)

    def test_the_packet_row_puts_each_value_in_its_column(self):
        blob = blobs.Blob((0.25, 0.5, 0.75), 0.2, (0.0, 0.0, 1.0), 7.0, swirl=0.5, shake=0.25)
        row = dict(zip(blobs.PACKET_COLUMNS, blobs.packet_row(blob, (16, 8, 4), 8.0, 1.5, DT)))
        self.assertEqual(len(row), 18)
        self.assertEqual((row['h'], row['extent_x'], row['extent_y'], row['extent_z']), (0.125, 2.0, 1.0, 0.5))
        self.assertEqual((row['centre_x'], row['centre_y'], row['centre_z']), (0.5, 0.5, 0.375))
        self.assertEqual((row['radius'], row['time'], row['dt'], row['amplitude']), (0.2, 1.5, DT, 7.0))
        self.assertEqual((row['swirl'], row['fluctuation'], row['direction_z']), (0.5, 0.25, 1.0))
        self.assertAlmostEqual(row['omega'], math.tau * blobs.SHAKE_HZ)

    def test_the_kernel_evaluation_matches_the_scenes_forcing(self):
        """forcing.py rebuilds recorded scenes' forcing independently; a blob with that scene's values is the same field."""
        scene = dict(counts=[12, 10, 8], cells_per_metre=16, radius=0.3, seed=7, frequency=1.3, acceleration=0.9,
                     swirl=0.6, fluctuation=0.4)
        centre, pulse, time_s = [0.4, 0.55, 0.6], 0.8, 0.35
        direction, phase = forcing._direction_and_phase(scene['seed'])
        extents = [n / 16 for n in scene['counts']]
        blob = blobs.Blob(tuple(centre), 0.3 * min(extents), tuple(direction), 0.9 * pulse, swirl=0.6, shake=0.4)
        row = blobs.packet_row(blob, scene['counts'], 16, time_s, DT, omega=math.tau * 1.3, phase=phase)
        expected = forcing._acceleration(scene, centre, pulse, time_s)
        for got, want in zip(blobs.kernel_increment(row, scene['counts']), expected):
            self.assertEqual(got.shape, want.shape)
            torch.testing.assert_close(got / DT, want, rtol=1e-12, atol=1e-12)
            self.assertGreater(float(want.abs().max()), 0.1)

    def test_one_face_by_the_kernels_own_arithmetic(self):
        """known_control_dynamic for one y face, written out as the CUDA source does it."""
        counts, cpm = (6, 5, 4), 4.0
        blob = blobs.Blob((0.4, 0.5, 0.6), 0.3, (0.6, 0.0, 0.8), 3.0, swirl=0.7, shake=0.5)
        row = blobs.packet_row(blob, counts, cpm, 0.2, DT)
        h, e, c, radius, omega, t, phase = row[0], row[1:4], row[4:7], row[7], row[8], row[9], row[10]
        direction, amplitude, swirl, fluctuation, dt = row[11:14], row[14], row[15], row[16], row[17]
        d, (qx, qy, qz) = 1, (2, 3, 1)
        pos = [(qx + 0.5) * h, qy * h, (qz + 0.5) * h]
        offsets = [(pos[a] - c[a]) / radius for a in range(3)]
        r2 = sum(o * o for o in offsets)
        wave = sum(math.sin(math.tau * pos[a] / e[a] + omega * t * (a + 1) + phase + d) for a in range(3)) / 3
        envelope = math.exp(-0.5 * r2) * (1 - r2 / 16) ** 2 if r2 < 16 else 0.0
        a, b = (d + 1) % 3, (d + 2) % 3
        rotation = direction[a] * offsets[b] - direction[b] * offsets[a]
        want = dt * (amplitude * envelope) * ((direction[d] + swirl * rotation) + fluctuation * wave)
        got = blobs.kernel_increment(row, counts)[1][qz, qy, qx].item()
        self.assertAlmostEqual(got, want, places=15)
        self.assertNotEqual(want, 0.0)

    def test_no_force_through_a_wall_and_forces_add(self):
        one, two = blobs.Blob.from_json(push()), blobs.Blob.from_json(push(centre=(0.2, 0.8, 0.5), direction=(1, 0, 0)))
        fx, fy, fz = blobs.increment([one, two], (16, 16, 16), 16.0, 0.0, DT)
        for field, axis in ((fx, 2), (fy, 1), (fz, 0)):
            self.assertEqual(float(field.select(axis, 0).abs().max()), 0.0)
            self.assertEqual(float(field.select(axis, -1).abs().max()), 0.0)
        separate = [sum(p) for p in zip(blobs.increment([one], (16, 16, 16), 16.0, 0.0, DT),
                                        blobs.increment([two], (16, 16, 16), 16.0, 0.0, DT))]
        for got, want in zip((fx, fy, fz), separate):
            torch.testing.assert_close(got, want)


def state_of(counts, u=None, v=None, w=None):
    nx, ny, nz = counts
    return dict(u=torch.zeros((nz, ny, nx + 1)) if u is None else u, v=torch.zeros((nz, ny + 1, nx)) if v is None else v,
                w=torch.zeros((nz + 1, ny, nx)) if w is None else w,
                pressure=torch.arange(nz * ny * nx, dtype=torch.float64).reshape(nz, ny, nx),
                tracer=torch.zeros((nz, ny, nx)))


class Planes(unittest.TestCase):
    def test_orientation_rows_are_the_vertical_axis(self):
        counts = (5, 4, 3)
        s = state_of(counts)
        nz, ny, nx = 3, 4, 5
        z_cut = views.plane(s, counts, 0.1, 'z', 1, 'pressure')
        self.assertEqual(tuple(z_cut.shape), (ny, nx))
        self.assertEqual(z_cut[2, 3].item(), s['pressure'][1, 2, 3].item())
        x_cut = views.plane(s, counts, 0.1, 'x', 2, 'pressure')        # rows y, columns z
        self.assertEqual(tuple(x_cut.shape), (ny, nz))
        self.assertEqual(x_cut[3, 1].item(), s['pressure'][1, 3, 2].item())
        y_cut = views.plane(s, counts, 0.1, 'y', 0, 'pressure')        # rows z, columns x
        self.assertEqual(tuple(y_cut.shape), (nz, nx))

    def test_velocity_fields_at_cell_centres(self):
        counts, h = (4, 4, 4), 0.25
        u = torch.ones((4, 4, 5)) * 3.0
        v = torch.zeros((4, 5, 4))
        v[:, :, :] = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)    # grows with y by 1 per face
        s = state_of(counts, u=u, v=v)
        self.assertTrue(torch.allclose(views.plane(s, counts, h, 'z', 2, 'vx'), torch.full((4, 4), 3.0)))
        self.assertTrue(torch.allclose(views.plane(s, counts, h, 'z', 2, 'vy')[:, 0], torch.tensor([.5, 1.5, 2.5, 3.5])))
        self.assertTrue(torch.allclose(views.plane(s, counts, h, 'z', 2, 'divergence'), torch.full((4, 4), 1 / h)))
        speed = views.plane(s, counts, h, 'z', 2, 'speed')
        self.assertAlmostEqual(speed[0, 0].item(), math.hypot(3.0, 0.5), places=5)
        for axis in 'xyz':
            for field in views.FIELDS:
                self.assertEqual(views.plane(s, counts, h, axis, 0, field).dtype, torch.float32)

    def test_a_solid_turn_has_its_vorticity(self):
        counts, h = (8, 8, 8), 0.5
        x = (torch.arange(8, dtype=torch.float32) + 0.5) * h
        u = (-(x.reshape(1, 8, 1) - 2)).expand(8, 8, 9).clone()      # u = -(y - c), v = x - c: vorticity 2 along z
        v = (x.reshape(1, 1, 8) - 2).expand(8, 9, 8).clone()
        omega = views.plane(state_of(counts, u=u, v=v), counts, h, 'z', 4, 'vorticity')
        self.assertTrue(torch.allclose(omega[2:-2, 2:-2], torch.full((4, 4), 2.0), atol=1e-5))

    def test_bad_views_are_refused(self):
        s = state_of((4, 4, 4))
        for axis, index, field in (('w', 0, 'speed'), ('z', 4, 'speed'), ('z', -1, 'speed'), ('z', 0, 'heat')):
            with self.assertRaises(ValueError):
                views.plane(s, (4, 4, 4), 0.25, axis, index, field)

    def test_a_frame_quantises_over_its_range(self):
        values = torch.tensor([[-2.0, 0.0], [1.0, 2.0]])
        encoded = frames.encode(values, 'vx')
        self.assertEqual((encoded['min'], encoded['max'], encoded['width'], encoded['height']), (-2.0, 2.0, 2, 2))
        self.assertEqual(list(base64.b64decode(encoded['data'])), [0, 128, 191, 255])
        self.assertEqual(frames.encode(torch.zeros(3, 3), 'speed')['max'], 0.0)
        a = frames.arrows(torch.ones(48, 48), torch.zeros(48, 48))
        self.assertEqual((a['block'], a['width'], a['max_speed']), (2, 24, 1.0))
        self.assertEqual(set(base64.b64decode(a['h'])), {127})


class FakeBackend(unittest.TestCase):
    def setUp(self):
        self.backend = backend_class('fake')()
        self.backend.reset(SceneSpec((16, 16, 16), 16.0, 'layer'))

    def test_the_interface(self):
        b = self.backend
        caps = b.capabilities()
        self.assertEqual((caps['id'], caps['kind'], caps['max_blobs']), ('fake', 'test', 8))
        self.assertIn([16, 16, 16], caps['grids'])
        self.assertEqual(b.info()['scene']['extents_m'], [1.0, 1.0, 1.0])
        with self.assertRaisesRegex(Unsupported, 'grids'):
            b.reset(SceneSpec((17, 16, 16), 16.0))
        with self.assertRaisesRegex(Unsupported, 'cells per metre'):
            b.reset(SceneSpec((16, 16, 16), 3.0))
        with self.assertRaisesRegex(Unsupported, 'at most 8'):
            b.set_forces([push()] * 9)
        b.set_forces([push()])
        for _ in range(3):
            b.step()
        self.assertEqual(b.info()['steps'], 3)
        for axis in 'xyz':
            for field in views.FIELDS:
                self.assertEqual(tuple(b.slice(axis, 8, field).shape), (16, 16))
        self.assertEqual(b.state['tracer'].shape, (16, 16, 16))

    def test_the_dye_moves_the_way_it_is_pushed(self):
        b = self.backend
        ys = (torch.arange(16, dtype=torch.float64) + 0.5).reshape(1, 16, 1)
        height = lambda: float((b.state['tracer'].double() * ys).sum() / b.state['tracer'].double().sum())
        start = height()
        b.set_forces([push(centre=(0.5, 0.3, 0.5), direction=(0, 1, 0), magnitude=30, radius=0.1)])
        for _ in range(30):
            b.step()
        self.assertGreater(height(), start + 0.05)
        b.reset_tracer('ball')
        self.assertTrue(torch.equal(b.state['tracer'], tracer_pattern((16, 16, 16), 'ball')))

    def test_the_listing_says_why_a_backend_cannot_run(self):
        rows = {row['id']: row for row in listing()}
        self.assertEqual(set(rows), {'composition', 'generator', 'reference', 'fake'})
        self.assertTrue(rows['fake']['available'])
        if not torch.cuda.is_available():
            self.assertFalse(rows['composition']['available'])
            self.assertIn('CUDA', rows['composition']['reason'])


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Pacing(unittest.TestCase):
    def test_steps_fall_due_every_sixtieth_of_a_second(self):
        p = Pacer(DT)
        self.assertEqual(p.delay(10.0), 0.0)
        self.assertFalse(p.stepped(10.004))
        self.assertAlmostEqual(p.delay(10.004), DT - 0.004)
        p.stepped(10.02)
        self.assertAlmostEqual(p.delay(10.02), 2 * DT - 0.02)
        self.assertEqual(p.delay(10.2), 0.0)                 # overdue: no sleep

    def test_falling_behind_drops_the_backlog(self):
        p = Pacer(DT, max_lag=0.25)
        p.delay(0.0)
        self.assertTrue(p.stepped(0.5))                       # one step took half a second
        self.assertEqual((p.start, p.count, p.late_at), (0.5, 0, 0.5))
        self.assertAlmostEqual(p.delay(0.5), 0.0)

    def test_stats_over_the_last_second(self):
        s = Stats(window=1.0)
        s.clear(0.0)
        for k in range(1, 121):                               # 120 steps over 2 s, 1..120 ms
            s.add(k / 60, float(k))
        out = s.summary(2.0)
        self.assertEqual(out['samples'], 61)                  # steps that ended in the last second
        self.assertAlmostEqual(out['median_ms'], 90.0)
        self.assertEqual(out['p95_ms'], 117.0)            # nearest rank: the 58th of 61
        self.assertAlmostEqual(out['steps_per_s'], 61.0)
        self.assertAlmostEqual(out['realtime_factor'], 61 / 60)
        self.assertAlmostEqual(out['headroom'], DT * 1000 / 90)
        self.assertEqual(s.summary(10.0)['samples'], 0)

    def test_the_thread_paces_pauses_and_single_steps(self):
        sim = Simulation(backend_class('fake')(), SceneSpec((9, 11, 11), 16.0), mode='realtime')
        sim.start()
        try:
            sim.control('pause')
            steps = sim.snapshot()['steps']
            sim.control('step', steps=3)
            for _ in range(100):
                if sim.snapshot()['steps'] == steps + 3:
                    break
                threading.Event().wait(0.01)
            self.assertEqual(sim.snapshot()['steps'], steps + 3)
            sim.control('mode', mode='fastest')
            sim.control('resume')
            threading.Event().wait(0.5)
            fast = sim.snapshot()
            self.assertGreater(fast['stats']['steps_per_s'], 60)
            self.assertFalse(fast['stats']['behind'])
            with self.assertRaises(ValueError):
                sim.control('mode', mode='slow')
            with self.assertRaises(ValueError):
                sim.control('fly')
        finally:
            sim.stop()


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation(backend_class('fake')(), SceneSpec((16, 16, 16), 16.0), mode='realtime')
        cls.sim.start()
        cls.server = serve(Playground(cls.sim), 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.sim.stop()

    def request(self, path, body=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        request = urllib.request.Request(self.origin + path, data=data, method='GET' if body is None else 'POST')
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                return response.status, (json.loads(raw) if response.headers['Content-Type'] == 'application/json' else raw)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_binds_the_loopback_address_only(self):
        self.assertEqual(self.server.server_address[0], '127.0.0.1')

    def test_page_and_capabilities(self):
        status, page = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn(b'Fluid playground', page)
        self.assertEqual(self.request('/static/view.js')[0], 200)
        self.assertEqual(self.request('/static/../server.py')[0], 404)
        self.assertEqual(self.request('/nothing')[0], 404)
        status, caps = self.request('/api/capabilities')
        self.assertEqual((status, caps['current']['id'], caps['modes']), (200, 'fake', ['realtime', 'fastest']))

    def test_frames_carry_planes_and_stats(self):
        status, frame = self.request('/api/frame?views=z::tracer,x:3:speed&arrows=1')
        self.assertEqual(status, 200)
        first, second = frame['views']
        self.assertEqual((first['axis'], first['index'], first['horizontal'], first['vertical']), ('z', 8, 'x', 'y'))
        self.assertEqual(len(base64.b64decode(first['data'])), 16 * 16)
        self.assertIn('arrows', first)
        self.assertNotIn('arrows', second)
        self.assertEqual((second['axis'], second['index']), ('x', 3))
        for key in ('median_ms', 'p95_ms', 'steps_per_s', 'realtime_factor', 'behind'):
            self.assertIn(key, frame['stats'])
        self.assertEqual(self.request('/api/frame?views=z:99:speed')[0], 400)

    def test_forces_are_kept_until_changed(self):
        status, body = self.request('/api/forces', dict(forces=[push(), push(direction=(1, 0, 0))]))
        self.assertEqual((status, len(body['forces'])), (200, 2))
        self.assertEqual(len(self.request('/api/frame')[1]['forces']), 2)
        status, body = self.request('/api/forces', dict(forces=[push(radius=-1)]))
        self.assertEqual(status, 400)
        self.assertIn('radius', body['error'])
        self.assertEqual(len(self.request('/api/frame')[1]['forces']), 2)
        self.assertEqual(self.request('/api/forces', dict(forces=[]))[1]['forces'], [])
        self.assertEqual(self.request('/api/forces', b'not json')[0], 400)
        self.assertEqual(self.request('/api/forces', dict(forces='all'))[0], 400)

    def test_control_reset_and_refusals(self):
        self.assertTrue(self.request('/api/control', dict(action='pause'))[1]['paused'])
        status, body = self.request('/api/control', dict(action='resume'))
        self.assertFalse(body['paused'])
        status, body = self.request('/api/reset', dict(counts=[32, 32, 32], tracer='stripes'))
        self.assertEqual((status, body['info']['scene']['counts'], body['info']['steps']), (200, [32, 32, 32], 0))
        status, body = self.request('/api/reset', dict(counts=[33, 32, 32]))
        self.assertEqual(status, 400)
        self.assertIn('runs only the grids', body['error'])
        self.assertEqual(self.request('/api/frame')[1]['status'], 'running')     # the refusal changed nothing
        self.assertEqual(self.request('/api/tracer', dict(pattern='ball'))[0], 200)
        self.assertEqual(self.request('/api/tracer', dict(pattern='paint'))[0], 400)
        status, body = self.request('/api/backend', dict(backend='nonesuch'))
        self.assertEqual(status, 400)
        if not torch.cuda.is_available():
            status, body = self.request('/api/backend', dict(backend='composition'))
            self.assertEqual(status, 400)
            self.assertIn('CUDA', body['error'])
        self.request('/api/reset', dict(counts=[16, 16, 16], tracer='layer'))


if __name__ == '__main__':
    unittest.main()
