"""Capture portable source and actual opened frozen model assets for managed fits."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from shared_temporal_generator_v3 import SharedTemporalGenerator
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only asset capture required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen asset source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = LongTemporalDataset(build_index([ROOT / p for p in reg['audits']]))
    scenes = [receipt['scene'] for _, receipt, _ in dataset.receipts.values() if receipt['split'] == 'train']
    if len(scenes) != 24:
        raise ValueError('Every training geometry required')
    args.output.mkdir(parents=True)
    opened = set()
    tracking = True

    def observe(event, values):
        if not tracking or event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        value = Path(os.fsdecode(values[0])).absolute()
        if value.is_relative_to(ROOT):
            relative = value.relative_to(ROOT)
            if 'private' in relative.parts:
                raise PermissionError('Model asset capture cannot open private credentials')
            if value.suffix in ('.npz', '.json', '.ptx', '.cu'):
                opened.add(value)

    sys.addaudithook(observe)
    model = None
    began = time.monotonic()
    try:
        model = SharedTemporalGenerator(reg['transport_pair'], reg['shared_pair'], reg['build'],
            reg['control_build'], hidden=4, force_sum_build=reg['force_sum_build'])
        before = inventory(model)
        with torch.inference_mode():
            for scene in scenes:
                sizes = (tuple(reversed(scene['counts'])), *dimensions(tuple(scene['counts'])))
                fields = {n: torch.zeros(s, device='cuda', dtype=torch.float64 if n == 'pressure' else torch.float32).flatten()
                          for n, s in zip(FIELDS, sizes)}
                outputs = model.predict(fields, scene, 0, 30, 4)
                if len(outputs) != 4 or any(not torch.isfinite(v).all() for output in outputs for v in output.values()):
                    raise ValueError('Nonfinite dummy asset-discovery forward')
                if (time.monotonic() - began > reg['maximum_wall_seconds']
                        or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                    raise ValueError('Finite asset-discovery budget exceeded')
        if inventory(model) != before:
            raise ValueError('Asset-discovery changed model parameters')
    finally:
        tracking = False
        if model is not None:
            model.close()
    sources = set()
    for directory in reg['source_directories']:
        sources.update(p for p in (ROOT / directory).rglob('*.py') if '__pycache__' not in p.parts)
    members = {str(p.relative_to(ROOT)): digest(p) for p in sources | opened}
    with zipfile.ZipFile(args.output / 'model-source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as captured:
        for name, expected in sorted(members.items()):
            if digest(ROOT / name) != expected:
                raise ValueError('Opened source/asset changed during capture')
            captured.write(ROOT / name, name)
    write(args.output / 'manifest.json', dict(schema='rheon.shared-temporal-source-assets.v1',
        members=members, asset_paths=[str(p.relative_to(ROOT)) for p in sorted(opened)],
        configuration={n: reg[n] for n in ('transport_pair', 'shared_pair', 'build', 'control_build', 'force_sum_build')}))
    write(args.output / 'results.json', dict(status='captured_shared_temporal_sources_and_actual_frozen_assets',
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        archive=dict(path=str(args.output / 'model-source.zip'), bytes=(args.output / 'model-source.zip').stat().st_size,
                     sha256=digest(args.output / 'model-source.zip')),
        manifest_sha256=digest(args.output / 'manifest.json'), source_files=len(sources), actual_opened_assets=len(opened),
        dummy_CUDA_forwards=24, native_accuracy_claim=False, cpu_model_forward=False,
        new_parameter_fits=0, optimizer_updates=0, reserved52_references_opened=0,
        portable_entrypoint_qualified=False, full_generator_target_pass=False,
        elapsed_seconds=time.monotonic() - began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
