"""Bounded reader for a captured24scene native training transport.

Only the explicitly supplied archive is opened. Decoding never follows native
receipt/NPZ paths. A separate qualified full36 eligibility proof is required
by a fitting entrypoint; this reader alone does not authorize a fit.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import zipfile

from lossless_training_array_codec_v1 import decode
from managed_input_stream_v1 import streamed_identity

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer', 'rhs')


class CapturedLongTrainingBank:
    def __init__(self, archive, expected_identity, cache_states=16):
        if streamed_identity(archive) != expected_identity:
            raise ValueError('Captured complete training archive differs')
        if type(cache_states) is not int or not 16 <= cache_states <= 32:
            raise ValueError('Finite decoded state cache required')
        self.archive = zipfile.ZipFile(archive)
        self.cache = OrderedDict()
        self.cache_states = cache_states
        self.payload_reads = 0
        self.payload_bytes_read = 0
        self.peak_cache_bytes = 0
        names = self.archive.namelist()
        if len(set(names)) != len(names) or 'manifest.json' not in names:
            raise ValueError('Unique framed member names required')
        if self.archive.getinfo('manifest.json').file_size > 32 * 1024**2:
            raise ValueError('Bounded native-state manifest required')
        self.manifest = json.loads(self.archive.read('manifest.json'))
        manifest = self.manifest
        if (manifest['schema'] != 'rheon.lossless-long-training-candidate.v1'
                or manifest['fields'] != list(FIELDS) or manifest['reset_interval'] != 16
                or len(manifest['masters']) != 24):
            raise ValueError('Exact full24 native training transport required')
        groups, expected_members = set(), {'manifest.json'}
        self.states, self.windows = {}, []
        for identifier, case in manifest['masters'].items():
            if case['split'] != 'train' or case['physical_scenario_group_id'] in groups:
                raise ValueError('Independent training masters only')
            groups.add(case['physical_scenario_group_id'])
            records = case['states']
            self.states[identifier] = {r['fine_step']: r for r in records}
            if len(self.states[identifier]) != len(records):
                raise ValueError('Unique physical state indices required')
            starts = sorted({w['start_fine_step'] for w in case['windows']})
            expected_windows = {(s, rate, k) for s in starts for rate in (30, 60, 120)
                                for k in (1, 2, 3, 4)}
            actual_windows = {(w['start_fine_step'], w['output_steps_per_s'], w['K'])
                              for w in case['windows']}
            if actual_windows != expected_windows or len(case['windows']) != len(expected_windows):
                raise ValueError('Every selected start requires all12 physical configurations')
            for ordinal, record in enumerate(records):
                previous = None if ordinal % 16 == 0 else records[ordinal - 1]['fine_step']
                if record['previous_fine_step'] != previous or set(record['arrays']) != set(FIELDS):
                    raise ValueError('Bounded canonical decoding chain required')
                for name, spec in record['arrays'].items():
                    member = f'states/{identifier}/{record["fine_step"]:05}/{name}.z'
                    if (spec['member'] != member or spec['xor_previous'] != (previous is not None)
                            or self.archive.getinfo(member).file_size != spec['compressed_bytes']):
                        raise ValueError('Typed captured array framing differs')
                    expected_members.add(member)
            for start in starts:
                examples = [w for w in case['windows'] if w['start_fine_step'] == start]
                required = {start}
                for w in examples:
                    targets = [start + j * (240 // w['output_steps_per_s']) for j in range(1, w['K'] + 1)]
                    if w['target_fine_steps'] != targets:
                        raise ValueError('Training head timing differs from whole output intervals')
                    required.update(targets)
                if not required <= set(self.states[identifier]):
                    raise ValueError('Complete current and allK future states required')
                self.windows.append(dict(identifier=identifier, start_fine_step=start,
                    configurations=examples, required_fine_steps=sorted(required),
                    scene_weight=1 / 24, window_weight=1 / len(starts)))
        if set(names) != expected_members:
            raise ValueError('Unexpected or missing captured training data members')

    def load(self, identifier, step):
        key = identifier, step
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        record = self.states[identifier][step]
        previous_step = record['previous_fine_step']
        previous = None if previous_step is None else self.load(identifier, previous_step)
        arrays = {}
        for name, spec in record['arrays'].items():
            payload = self.archive.read(spec['member'])
            if hashlib.sha256(payload).hexdigest() != spec['payload_sha256']:
                raise ValueError('Captured compressed payload changed')
            self.payload_reads += 1
            self.payload_bytes_read += len(payload)
            arrays[name] = decode(spec, payload, None if previous is None else previous[name])
        self.cache[key] = arrays
        while len(self.cache) > self.cache_states:
            self.cache.popitem(last=False)
        self.peak_cache_bytes = max(self.peak_cache_bytes,
            sum(a.nbytes for state in self.cache.values() for a in state.values()))
        return arrays

    def cuda_window(self, window):
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA-only training window transfer')
        identifier = window['identifier']
        case = self.manifest['masters'][identifier]
        targets = {step: {n: torch.from_numpy(a).to('cuda') for n, a in self.load(identifier, step).items()}
                   for step in window['required_fine_steps']}
        current = {n: targets[window['start_fine_step']][n] for n in FIELDS[:-1]}
        return dict(scene=case['scene'], current=current, targets=targets)

    def close(self):
        self.cache.clear()
        self.archive.close()
