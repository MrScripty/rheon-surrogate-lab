"""Stream-check an existing training view without model imports or forwards.

Retain the producer's observed memory overrun; artifact integrity does not
retroactively make its resource qualification pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import zipfile


CHUNK_BYTES = 1024 * 1024


def streamed_digest(handle):
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Require a fresh audit output')
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads(args.protocol.read_text())
    for name, expected in protocol['source_sha256'].items():
        with (root / name).open('rb') as handle:
            if streamed_digest(handle)[1] != expected:
                raise ValueError('Frozen source changed: ' + name)
    manifest_path = root / protocol['manifest']
    manifest = json.loads(manifest_path.read_text())
    archive = manifest_path.parent / manifest['archive']['path']
    with archive.open('rb') as handle:
        size, sha = streamed_digest(handle)
    if (size, sha) != (manifest['archive']['bytes'], manifest['archive']['sha256']):
        raise ValueError('Archive identity changed')
    parts = [manifest['sparse_part'], *manifest['geometry_parts'], *manifest['parts']]
    names = [part['member'] for part in parts]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate declared archive member')
    with zipfile.ZipFile(archive) as captured:
        if len(captured.infolist()) != len(names) or set(captured.namelist()) != set(names):
            raise ValueError('Archive inventory differs')
        for part in parts:
            with captured.open(part['member']) as handle:
                measured = streamed_digest(handle)
            if measured != (part['bytes'], part['sha256']):
                raise ValueError('Archive member changed: ' + part['member'])
    if len(manifest['parts']) != 783 or len(manifest['control_rows']) != 1071 or len(manifest['scene_rows']) != 148:
        raise ValueError('Incomplete training view')
    producer_protocol = json.loads((root / protocol['producer_protocol']).read_text())
    limit = producer_protocol['maximum_peak_rss_bytes']
    observed = manifest['observed_peak_rss_bytes']
    result = dict(
        status='verified_training_artifact_integrity_producer_resource_gate_failed',
        manifest=protocol['manifest'], archive_bytes=size, archive_sha256=sha,
        verified_members=len(parts), local_samples=783, fixed_native_controls=1071,
        training_scenes=148, producer_peak_rss_bytes=observed,
        producer_maximum_peak_rss_bytes=limit, producer_resource_gate_pass=observed <= limit,
        producer_overrun_cause='Final archive digest uses Path.read_bytes through frozen support.digest; original output retained.',
        stream_chunk_bytes=CHUNK_BYTES, audit_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        elapsed_seconds=time.monotonic() - started,
        new_parameter_fits=0, cpu_model_forward=False, expanded_reserved52_references_opened=0,
        native_accuracy_claim=False, latency_claim=False, full_generator_target_pass=False,
    )
    if result['audit_peak_rss_bytes'] > protocol['maximum_peak_rss_bytes']:
        raise ValueError('Streaming audit exceeded memory budget')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
