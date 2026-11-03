"""Verify archived CPU reproducibility artifacts; never import/run a model."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def run(args):
    if args.output.exists():raise ValueError('Fresh snapshot integrity output required')
    manifest=json.loads(args.manifest.read_text());snapshot=ROOT/manifest['snapshot_repo']
    if manifest['status']!='completed_cpu_backend_reproduction_snapshot':raise ValueError('Completed snapshot required')
    total=0
    for path,spec in manifest['files'].items():
        value=snapshot/path
        if value.stat().st_size!=spec['bytes'] or digest(value)!=spec['sha256']:
            raise ValueError('Archived CPU artifact changed: '+path)
        total+=spec['bytes']
    if total!=manifest['total_bytes']:raise ValueError('Complete archived population required')
    protocol=snapshot/'docs/plans/full-ai-data-diversity-v1/divisor-prototype-benchmark-protocols-01/prototype-benchmark-p1.json'
    reg=json.loads(protocol.read_text())
    for path,expected in reg['source_sha256'].items():
        if manifest['files'][path]['sha256']!=expected:raise ValueError('Frozen CPU benchmark binding is not archived')
    runtime=json.loads((snapshot/'runs/dependency-migration-v1/runtime/current-manifest-v2.json').read_text())
    if not Path(runtime['python']).is_file() or not Path(runtime['site_packages']).is_dir():
        raise ValueError('Pinned external managed Python installation is unavailable')
    if digest(reg['library_path'])!=reg['library_sha256']:raise ValueError('Pinned external FFTW changed')
    result=dict(status='passed_cpu_snapshot_integrity_without_inference',created_unix=time.time(),
        snapshot_manifest_sha256=digest(args.manifest),files_verified=len(manifest['files']),bytes_verified=total,
        cpu_benchmark_source_bindings_verified=len(reg['source_sha256']),
        archived_cpu_runtime_manifest_sha256=digest(snapshot/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'),
        external_runtime_python_exists=True,external_runtime_packages_directory_exists=True,
        external_runtime_all_files_rehashed=False,external_fftw_library_hash_verified=True,
        cpu_model_forward_executed=False,torch_imported=False,optimizer_updates=0,
        timing_reproduction_claim=False,new_cpu_hardware_compatibility_claim=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('CPU snapshot integrity passed:',result['files_verified'],'files; no model inference.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
