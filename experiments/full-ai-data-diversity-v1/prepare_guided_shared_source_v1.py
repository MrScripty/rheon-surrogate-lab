"""Package new scientific sources with previously captured exact frozen assets.

This preparation performs no model forward or fit. The later isolated CUDA
qualification must prove actual constructors and all physical cases require no
ambient scientific reads; inherited assets are not claimed newly discovered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Fresh guided source package required')
    reg = json.loads(args.protocol.read_text())
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen package source differs: ' + name)
    prior = json.loads((ROOT / reg['inherited_manifest']).read_text())
    sources = {p for directory in reg['source_directories']
               for p in (ROOT / directory).rglob('*.py') if '__pycache__' not in p.parts}
    assets = {ROOT / name for name in prior['asset_paths']}
    if any(digest(p) != prior['members'][str(p.relative_to(ROOT))] for p in assets):
        raise ValueError('Previously captured frozen asset changed')
    members = {str(p.relative_to(ROOT)): digest(p) for p in sources | assets}
    if any('private' in Path(name).parts for name in members):
        raise ValueError('Private data cannot enter scientific source packages')
    if sum((ROOT / name).stat().st_size for name in members) > reg['maximum_uncompressed_bytes']:
        raise ValueError('Guided source package exceeds finite size budget')
    args.output.mkdir(parents=True)
    archive = args.output / 'model-source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as result:
        for name, expected in sorted(members.items()):
            if digest(ROOT / name) != expected:
                raise ValueError('Scientific source changed while packaging')
            result.write(ROOT / name, name)
    manifest = dict(schema='rheon.shared-temporal-source-assets.v1', members=members,
        asset_paths=prior['asset_paths'], configuration=prior['configuration'],
        model_family='guided_shared_residual_v2',
        inherited_actual_asset_capture=reg['inherited_manifest'],
        inherited_actual_asset_capture_sha256=digest(ROOT / reg['inherited_manifest']))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    record = dict(status='prepared_guided_sources_with_exact_inherited_frozen_assets',
        created_unix=time.time(), source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        archive=dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest(archive)),
        manifest_sha256=digest(args.output / 'manifest.json'), source_files=len(sources),
        inherited_frozen_assets=len(assets), model_forwards=0, new_parameter_fits=0,
        native_future_fields_read=0, reserved52_references_opened=0,
        actual_portable_constructor_forward_backward_qualified=False,
        next='Isolated captured CUDA full24/all12 zero-head byte parity and finite gradients, no parameter updates.')
    (args.output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    print(record['status'], len(sources), len(assets), archive.stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
