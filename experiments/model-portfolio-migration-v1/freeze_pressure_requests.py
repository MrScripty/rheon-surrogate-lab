"""Bind the already-scored pressure population to native cases and old captures.

This is migration benchmark input, not a second scientific registry. Native
case IDs and consumed exposure are retained. No model, tensor runtime, solver
or new reference is executed. Full-state/temporal populations remain separate.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from run_pressure_portfolio_v2 import verify_material

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/model-portfolio-migration-v1'
DOCS = ROOT / 'docs/plans/model-portfolio-migration-v1'
PACKETS = ROOT / 'runs/water-early-pressure-v1/control/project/inputs'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind_cases(manifest, native_cases):
    require(manifest['population'] == 'validation', 'Original consumed validation population required')
    rows = []
    seen = set()
    for index, item in enumerate(native_cases):
        original, dims = item['original_case'], item['dimensions']
        parent = manifest['parents'][original['parent']]
        entry = parent['entry']
        key = original['parent'], original['native_frame']
        require(key not in seen and item['id'] == f'case-{index:03d}', 'Native case identity/order differs')
        seen.add(key)
        frames = [f for f in parent['frames'] if f['index'] == original['native_frame']]
        require(len(frames) == 1, 'Original pressure frame is absent or ambiguous')
        require(entry['split'] == 'validation' and entry['trajectory_id'] == original['trajectory_id'],
                'Original trajectory/exposure differs')
        counts = parent['counts_xyz']
        require(counts == original['counts_xyz'] == parent['scene']['counts_xyz']
                and counts == [dims['grid_x'], dims['grid_y'], dims['grid_z']]
                and all(type(n) is int and n > 0 for n in counts), 'Original grid dimensions differ')
        spacing, density, dt = parent['cell_size_m'], parent['density_kg_m3'], entry['dt_s']
        require(all(math.isfinite(v) and v > 0 for v in (spacing, density, dt)), 'Positive finite physical metadata required')
        require(spacing == original['cell_size_m'] and 1./spacing == dims['cells_per_metre']
                and math.isclose(parent['scene']['cells_per_metre'], 1./spacing, rel_tol=1e-12)
                and density == dims['fluid_density_kg_m3'] and dt == dims['input_dt_s'],
                'Native case spacing, density or timestep differs from its original capture')
        for field, value in (('cell_size_m', spacing), ('rho', density), ('dt', dt)):
            require(hashlib.sha256(struct.pack('<d', value)).hexdigest() == original['input_sha256'][field],
                    'Physical scalar differs from its captured input tensor: '+field)
        for field, value in frames[0]['sha256'].items():
            if field in ('base', 'native_input_residual'):
                wanted = original['input_sha256'][field]
            elif field in ('reference', 'rhs'):
                wanted = original['reference_pressure_sha256' if field == 'reference' else 'reference_rhs_sha256']
            else:
                raise ValueError('Unexpected captured pressure field')
            require(value == wanted, 'Original field identity differs from its native case: '+field)
        rows.append(dict(id=item['id'], dimensions=dims, original_case=original,
                         capture_manifest_sha256=parent['capture_manifest_sha256'],
                         counts_xyz=counts, voxels=math.prod(counts), cells_per_metre=1./spacing,
                         input_dt_s=dt, fluid_density_kg_m3=density,
                         scene=parent['scene'], source_population=parent['source'],
                         physical_timestamp_s=None,
                         clock_note='Native frame index is retained; no physical timestamp is reconstructed.'))
    require(len(rows) == sum(len(p['frames']) for p in manifest['parents']), 'Native cases omit original captured frames')
    return rows


def freeze(output, report):
    require(not output.exists() and not report.exists(), 'Fresh benchmark contract and report required')
    plan_path = DOCS/'reports/pressure-common-score-plan-01.json'
    plan = json.loads(plan_path.read_bytes())
    manifest_path = PACKETS/'validation_splits.json'
    material_path = ROOT/plan['material']
    for path in (manifest_path, material_path):
        require(sha(path) == plan['source_sha256'][str(path.relative_to(ROOT))], 'Previously scored input identity changed')
    manifest = json.loads(manifest_path.read_bytes())
    for name, shard in zip(('validation_dataset.npz', 'validation_dataset_extra.npz'), manifest['shards'], strict=True):
        path = PACKETS/name
        require(path.stat().st_size == shard['bytes'] and sha(path) == shard['sha256'], 'Original pressure packet changed')
    bundle_path = RUN/'native-pressure-portfolio-bundle-01/bundle.json'
    bundle = json.loads(bundle_path.read_bytes())
    packet_info = bundle['packets'][0]
    packet_path = bundle_path.parent/packet_info['path']
    require(sha(packet_path) == packet_info['sha256'], 'Original native pressure packet changed')
    packet = json.loads(packet_path.read_bytes())
    context_source = next(s for s in packet['sources'] if s['id'] == 'context')
    context_raw = base64.b64decode(context_source['data_base64'], validate=True)
    require(len(context_raw) == context_source['bytes']
            and hashlib.sha256(context_raw).hexdigest() == context_source['sha256'], 'Native context bytes changed')
    context = json.loads(context_raw)
    cases = bind_cases(manifest, context['cases'])
    require(len(cases) == 128 and [c['id'] for c in packet['cases']] == [c['id'] for c in cases], 'All128 native cases required')
    captures = {}
    for parent in manifest['parents']:
        path = Path(parent['capture_root'])/'manifest.json'
        require(path.resolve().is_relative_to(ROOT) and sha(path) == parent['capture_manifest_sha256'], 'Original capture manifest unavailable or changed')
        capture = json.loads(path.read_bytes())
        captures[parent['capture_manifest_sha256']] = dict(path=str(path.relative_to(ROOT)),
            sha256=parent['capture_manifest_sha256'], native_build_receipt=capture['native_build_receipt'])
    material = json.loads(material_path.read_bytes())
    require(len(material['models']) == 127, 'All127 available named checkpoints required')
    verify_material(material)
    models = []
    for entry in material['models']:
        path = ROOT/entry['restore']/'weights.npz'
        require(sha(path) == entry['weights_sha256'], 'Original pressure checkpoint weight bytes changed')
        models.append(dict(identifier=entry['id'], weights_sha256=entry['weights_sha256'],
                           restore=entry['restore'], fit_item_id=entry['fit_item_id']))
    contract = dict(schema='rheon.frozen-pressure-request-benchmark.v1', task='pressure_proposal',
        scope_id=packet['scope']['id'], exposure='consumed; not independent confirmation',
        native_case_binding_sha256=bundle['case_binding_sha256'], native_context_sha256=context_source['sha256'],
        cases=cases, original_captures=list(captures.values()), models=models,
        unavailable=['C01','C02'], inference_inputs=['base','native_input_residual','cell_size_m','rho','dt'],
        scoring_only_fields=['reference','rhs'], pressure_dtype='float64',
        output='One pressure proposal tensor for the supplied request; no complete simulation advance.',
        simulation_steps_per_call=None, physical_timestamp_inferred=False,
        primary_timing=dict(device='GPU', metric='complete_pressure_call_ms',
            boundary='Complete declared CPU input tensors to independently owned CPU pressure, including transfers and synchronization.',
            setup_and_warmups='separate', denominator='one pressure request; never K internal or physical simulation steps'),
        common_score_plan=dict(path=str(plan_path.relative_to(ROOT)), sha256=sha(plan_path)),
        packet_manifest=dict(path=str(manifest_path.relative_to(ROOT)), sha256=sha(manifest_path)),
        material=dict(path=str(material_path.relative_to(ROOT)), sha256=sha(material_path)),
        native_packet=dict(path=str(packet_path.relative_to(ROOT)), sha256=sha(packet_path)))
    with output.open('x') as stream:
        json.dump(contract, stream, indent=2, allow_nan=False);stream.write('\n')
    summary = dict(schema=contract['schema'], status='frozen_existing_consumed_pressure_population',
        created_unix=time.time(), contract=dict(path=str(output.relative_to(ROOT)), sha256=sha(output)),
        cases=len(cases), named_weights_reverified=len(models), original_capture_manifests=len(captures),
        original_reference_binaries=sorted({c['native_build_receipt']['binary_sha256'] for c in captures.values()}),
        by_cells_per_metre=dict(Counter(str(c['cells_per_metre']) for c in cases)),
        input_timesteps_s=sorted({c['input_dt_s'] for c in cases}),
        model_forwards=0, new_references=0, latency_claim=False, full_portfolio_benchmark_complete=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__), ROOT/'experiments/model-portfolio-migration-v1/run_pressure_portfolio_v2.py')})
    with report.open('x') as stream:
        json.dump(summary, stream, indent=2);stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    freeze(args.output.resolve(), args.report.resolve())
