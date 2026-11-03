"""Prepare auditable lossless views; the original qualified packets remain immutable."""
import io
from pathlib import Path
import numpy as np
from policy import LAB, MIB, canonical, decode, fresh, load_module, read, require, sha, write
import data


def create(sources, selections, population, output):
    # The original loader proves source numeric inventories and physical reservation.
    original = load_module('sequence_original_water_reader', LAB/'experiments/sealed-water-v1/data.py')
    authorities = {}; loaded = {}; parents = []; values = {}
    for name, packet in sources.items():
        raw = read(packet/'dataset.npz', 32*MIB); manifest_raw = read(packet/'splits.json', 4*MIB)
        manifest = decode(manifest_raw); reservation, records = original.load(raw, manifest)
        receipt = decode(read(packet/'dataset-receipt.json', MIB))
        require(receipt['status'] == 'verified_lossless_water' and receipt['dataset_sha256'] == sha(raw) and receipt['manifest_sha256'] == sha(manifest_raw), 'source_packet', 'qualified original lossless packet')
        require(len({p['capture_manifest_sha256'] for p in records}) == 1, 'source_capture', 'one qualified native capture')
        authorities[name] = {'reservation': reservation, 'capture_manifest_sha256': records[0]['capture_manifest_sha256'], 'packet_manifest_sha256': sha(manifest_raw)}
        loaded[name] = records
    for source, split, ordinal, slots in selections:
        choices = [p for p in loaded[source] if p['entry']['split'] == split]
        parent = choices[ordinal]; selected = [parent['frames'][slot] for slot in slots]
        index = len(parents)
        for field in data.FIELDS:
            values[f'p{index:04d}_{field}'] = parent['arrays'][field][slots].copy()
        parents.append({'source': source, 'scene': parent['scene'],
            'entry': {**parent['entry'], 'trajectory_id': source+'--'+parent['entry']['trajectory_id']},
            **{key: parent[key] for key in ('counts_xyz', 'cell_size_m', 'density_kg_m3', 'capture_root', 'capture_manifest_sha256')}, 'frames': selected})
    require(len(parents) >= 1, 'view_selection', 'nonempty complete simulation selection')
    # Keep each input independently below the existing 32 MiB decoder bound.
    groups = [[], []]; sizes = [0, 0]
    for key, value in values.items():
        group = int(sizes[1] < sizes[0]); groups[group].append(key); sizes[group] += value.nbytes
    require(all(groups), 'view_shards', 'both bounded shards have arrays')
    shards = []; payloads = []
    for keys in groups:
        buffer = io.BytesIO(); np.savez(buffer, **{key: values[key] for key in keys}); raw = buffer.getvalue()
        require(len(raw) <= 32*MIB, 'view_shards', 'per-input size limit')
        payloads.append(raw); shards.append({'sha256': sha(raw), 'bytes': len(raw), 'keys': keys})
    reservation = {'schema': 'rheon-water-learning-reservation-v1', 'sources': authorities,
        'selection': [{'source': p['source'], 'trajectory_id': p['scene']['trajectory_id'], 'frames': [f['index'] for f in p['frames']]} for p in parents]}
    manifest = {'schema': 'rheon-water-learning-view-v1', 'reservation': reservation, 'reservation_sha256': sha(canonical(reservation)),
        'shards': shards, 'parents': parents, 'arrays': data.inventory(values), 'population': population}
    manifest_raw = canonical(manifest); require(len(manifest_raw) <= 4*MIB, 'view_manifest', 'metadata size limit')
    _reservation, reopened = data.load(tuple(payloads), manifest)
    require(all(np.array_equal(reopened[i]['arrays'][field], values[f'p{i:04d}_{field}']) for i in range(len(parents)) for field in data.FIELDS), 'view_parity', 'every original selected numeric value')
    output = fresh(output)
    for name, raw in zip(('dataset.npz', 'dataset-extra.npz'), payloads): (output/name).write_bytes(raw)
    (output/'splits.json').write_bytes(manifest_raw)
    receipt = {'status': 'verified_lossless_water_view', 'reader_source_sha256': sha(read(Path(data.__file__))),
        'parents': len(parents), 'frames': sum(len(p['frames']) for p in parents), 'population': population,
        'inputs': {slot: {'bytes': len(raw), 'sha256': sha(raw)} for slot, raw in data.inputs(output).items()}}
    write(output/'dataset-receipt.json', receipt)
    return receipt
