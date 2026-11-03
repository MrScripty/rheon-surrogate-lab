"""Capture teacher code, restored assets and training current states without future arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

from managed_input_stream_v1 import streamed_identity
from capture_memory_telemetry_v1 import process_memory

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def run(args):
    reg = read(args.protocol)
    if args.output.exists():
        raise ValueError('Fresh immutable compact source directory required')
    for p, expected in reg['source_sha256'].items():
        if streamed_identity(ROOT / p)['sha256'] != expected:
            raise ValueError('Frozen packing dependency changed: ' + p)
    source_reg = read(ROOT / reg['view_protocol'])
    # The qualified full view is the reference; no new scene or teacher is chosen.
    view_path = ROOT / reg['full_manifest']
    view = read(view_path)
    bank = read(ROOT / source_reg['bank'])
    if bank['split'] != 'train' or len(bank['trajectories']) != 148 or view['development_or_reserved_used']:
        raise ValueError('Original complete training reservation required')
    active = [c for c in bank['trajectories'] if c['reference_substeps'] == 8]
    if len(active) != 29:
        raise ValueError('All29active scenes required')
    paths, expected = {}, {}
    def add(path, digest=None):
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(ROOT)
        if '..' in path.parts or path.is_symlink() or not (ROOT / path).is_file():
            raise ValueError('Concrete owned file required: ' + str(path))
        identity = streamed_identity(ROOT / path)
        if digest is not None and identity['sha256'] != digest:
            raise ValueError('Frozen asset/current state changed: ' + str(path))
        paths[str(path)] = ROOT / path
        expected[str(path)] = identity
    cases = []
    for case in active:
        add(case['receipt'], case['receipt_sha256'])
        states = [s for s in case['states'] if s['step'] in view['starts']]
        if len(states) != 9:
            raise ValueError('All9current states required')
        for state in states:
            add(state['path'], state['sha256'])
        cases.append(dict(identifier=case['identifier'], receipt=case['receipt'],
            receipt_sha256=case['receipt_sha256'], reference_substeps=8, states=states))
    for name in ('manifest.json', 'network-normalization.json', 'optimization-normalization.json'):
        add(view_path.parent / name)
    add(view['original_eval_normalization'])
    # Constructors retain exact managed provenance, even for an inherited modal
    # model that is replaced by the shared response before any learned forward.
    pairs = [source_reg['transport_pair'], source_reg['shared_pair'],
        'runs/full-ai-next-priorities-v1/b1-fixed-viscosity-pair-01',
        'runs/full-ai-tight-targets-v1/mixed-modal-pair-11']
    for relative in pairs:
        pair = ROOT / relative
        receipt = read(pair / 'receipt.json')
        for path in (pair / 'receipt.json', pair / 'restore/receipt.json', pair / 'restore/weights.npz'):
            add(path)
        for key in ('training_run_id', 'restore_run_id'):
            add(pair / ('terminal-' + receipt[key] + '.json'))
        if relative.endswith('mixed-modal-pair-11'):
            assembly_path = ROOT / receipt['source_project'] / 'assembly.json'
            add(assembly_path, receipt['source_assembly_sha256'])
            add(read(assembly_path)['static_inputs']['manifest']['path'])
    builds = [source_reg['build'], source_reg['control_build'],
        'runs/full-ai-data-diversity-v1/debug/cube128-equation-failure-01/dct-debug-build-02',
        'runs/full-ai-data-diversity-v1/debug/cube64-density100-failure-01/dct-build-01']
    for name in builds:
        build = ROOT / name
        for path in build.iterdir():
            if path.is_file() and path.suffix in ('.json', '.ptx', '.cu'):
                add(path)
        receipt = read(build / 'receipt.json')
        if isinstance(receipt.get('source'), str):
            add(receipt['source'], receipt.get('source_sha256'))
    code = {}
    for p in sorted((ROOT / 'experiments').rglob('*.py')):
        if '__pycache__' in p.parts or p.is_symlink():
            continue
        relative = str(p.relative_to(ROOT))
        identity = streamed_identity(p)
        if relative in source_reg['source_sha256'] and identity['sha256'] != source_reg['source_sha256'][relative]:
            raise ValueError('Qualified teacher source changed: ' + relative)
        code[relative] = dict(path=p, **identity)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    def bounded():
        if (process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes'] or
                time.monotonic() - started > reg['maximum_wall_seconds'] or
                sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()) > reg['maximum_output_bytes']):
            raise ValueError('Compact source packing resource budget exceeded')
    def write_files(path, entries, compression):
        with zipfile.ZipFile(path, 'x', compression=compression, allowZip64=True) as archive:
            for relative, entry in entries.items():
                archive.write(entry['path'], relative)
        with zipfile.ZipFile(path) as archive:
            for relative, entry in entries.items():
                digest = hashlib.sha256()
                with archive.open(relative) as handle:
                    for chunk in iter(lambda: handle.read(1024**2), b''):
                        digest.update(chunk)
                if archive.getinfo(relative).file_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                    raise ValueError('Captured compact member changed: ' + relative)
        bounded()
    write_files(args.output / 'teacher-code.zip', code, zipfile.ZIP_DEFLATED)
    if (args.output / 'teacher-code.zip').stat().st_size > 8 * 1024**2:
        raise ValueError('Code must fit the existing declared source-file budget')
    inputs = {name: dict(path=path, **expected[name]) for name, path in paths.items()}
    write_files(args.output / 'teacher-inputs.zip', inputs, zipfile.ZIP_STORED)
    # Small exported learned geometry/response arrays support the unchanged
    # portable training loss without carrying all precomputed target fields.
    assets = {}
    with zipfile.ZipFile(view_path.parent / view['archive']['path']) as original, zipfile.ZipFile(args.output / 'loss-assets.zip', 'x', compression=zipfile.ZIP_STORED) as captured:
        for spec in [view['sparse_part'], *view['geometry_parts']]:
            raw = original.read(spec['member'])
            if len(raw) != spec['bytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
                raise ValueError('Exported learned loss asset changed')
            captured.writestr(spec['member'], raw)
            assets[spec['member']] = {k: spec[k] for k in ('bytes', 'sha256', 'arrays')}
    config = dict(format='compact_internal_teacher_source_v1', training_scenes=148,
        active_scenes=29, active_current_states=261, fixed_control_scenes=119,
        fixed_control_windows=1071, starts=view['starts'], cases=cases,
        original_manifest=str(view_path.relative_to(ROOT)), teacher_configuration={k: source_reg[k] for k in
            ('transport_pair', 'shared_pair', 'build', 'control_build', 'maximum_additive_closure_error')},
        geometry_and_sparse_assets=assets, source_code={n: {k:v[k] for k in ('bytes','sha256')} for n,v in code.items()},
        captured_inputs=expected, future_native_state_arrays_included=False,
        precomputed_active_future_labels_included=False, development_or_reserved_used=False,
        regeneration_qualified=False, fitting_allowed=False)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    bounded()
    outputs={p.name:streamed_identity(p) for p in args.output.iterdir() if p.is_file()}
    result=dict(status='captured_compact_teacher_source_regeneration_not_yet_qualified',
        created_unix=time.time(), outputs=outputs, code_files=len(code), input_files=len(inputs),
        active_current_states=261, active_current_snapshot_bytes=sum(Path(ROOT/s['path']).stat().st_size for c in cases for s in c['states']),
        all148scene_objective_metadata_preserved=True, future_native_state_arrays_included=False,
        precomputed_active_future_labels_included=False, development_or_reserved_used=False,
        all_captured_member_identities_verified=True, regeneration_qualified=False, fitting_allowed=False,
        optimizer_updates=0,new_parameter_fits=0,cpu_model_forward=False,
        expanded_reserved52_references_opened=0,full_generator_target_pass=False,
        memory_counters=process_memory(),elapsed_seconds=time.monotonic()-started,
        source_sha256=reg['source_sha256'],protocol_sha256=streamed_identity(args.protocol)['sha256'])
    (args.output / 'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],outputs=outputs)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
