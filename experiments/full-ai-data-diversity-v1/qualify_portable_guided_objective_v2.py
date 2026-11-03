"""Fresh portable CUDA entrypoint: captured source/assets and complete train bank.

No optimizer updates. Imported model code resolves against the extracted source
root; original repository scientific reads and native archive fallback are
denied during actual constructor/forward/backward execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    original_root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only portable objective qualification required')
    reg = json.loads(args.protocol.read_text())
    for name, expected in reg['source_sha256'].items():
        if sha(original_root / name) != expected:
            raise ValueError('Portable source closure changed: ' + name)
    source_manifest = json.loads((original_root / reg['source_manifest']).read_text())
    source_archive = original_root / reg['source_archive']
    transport = json.loads((original_root / reg['transport_results']).read_text())
    normalization = json.loads((original_root / reg['normalization']).read_text())
    output.mkdir(parents=True)
    result = dict(status='running', source_sha256=reg['source_sha256'],
        inference_device='cuda', cpu_model_forward=False, new_parameter_fits=0,
        optimizer_updates=0, reserved52_references_opened=0, rows=[],
        full_generator_target_pass=False, managed_run=False)
    began = time.monotonic()
    temporary = tempfile.TemporaryDirectory(prefix='rheon-portable-shared-')
    root = Path(temporary.name)
    with zipfile.ZipFile(source_archive) as captured:
        if set(captured.namelist()) != set(source_manifest['members']):
            raise ValueError('Captured source membership differs')
        for name, expected in source_manifest['members'].items():
            target = root / name
            if not target.resolve().is_relative_to(root):
                raise ValueError('Captured source path traversal')
            value = captured.read(name)
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError('Captured source/asset bytes differ')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    allowed_runtime = original_root / 'runs/dependency-migration-v1/app-env'
    guarded, denied = True, []

    def restrict(event, values):
        if not guarded or event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).absolute()
        if ((path.is_relative_to(original_root) and not path.is_relative_to(allowed_runtime)
                and not path.is_relative_to(output))
                or path.is_relative_to('/tmp/rheon-surrogate-lab-long-reference-240-v1')):
            denied.append(str(path))
            raise PermissionError('Portable model cannot read ambient scientific files')

    # Runtime packages remain installed dependencies; scientific Python imports
    # must come from the captured root instead of the current working tree.
    sys.path = [p for p in sys.path if not p or not Path(p).absolute().is_relative_to(original_root)
                or Path(p).absolute().is_relative_to(allowed_runtime)]
    sys.path[:0] = [str(root / 'experiments' / n) for n in
        ('full-ai-data-diversity-v1', 'full-ai-tight-targets-v1', 'full-ai-next-priorities-v1', 'pressure-methods-v2')]
    os.chdir(root)
    sys.addaudithook(restrict)
    bank = model = None
    try:
        import torch
        from guided_shared_fit_core_v2 import make_model, full_objective
        from captured_long_training_bank_v1 import CapturedLongTrainingBank
        from captured_parameter_inventory_v1 import inventory
        os.sched_setaffinity(0, {reg['host_cpu']})
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        archive = transport['archive']
        bank = CapturedLongTrainingBank(archive['path'], {n: archive[n] for n in ('bytes', 'sha256')})

        def bounded():
            if (time.monotonic() - began > reg['maximum_wall_seconds']
                    or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                raise ValueError('Registered portable objective resource budget exceeded')

        for width in reg['hidden_widths']:
            model = make_model(source_manifest['configuration'], width)
            before = inventory(model)
            scored = full_objective(model, bank, normalization, True, bounded,
                lambda n, count, value: print('Portable guided full24 zero-head physical objective:', width, n, '/', count, flush=True) if n % 24 == 0 else None, certify_zero_heads=True)
            if inventory(model) != before:
                raise ValueError('Backward-only qualification changed model parameters')
            result['rows'].append(dict(hidden_width=width, parameter_inventory=before,
                neural_parameters=sum(p.numel() for p in model.network.parameters()), **scored))
            (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
            model.close()
            model = None
        if denied:
            raise ValueError('Portable scientific operation attempted ambient reads')
        result.update(status='passed_portable_guided_full24_all12_zero_heads_and_gradients',
            unexpected_ambient_reads=0, source_assets_extracted_and_verified=True,
            captured_train_data_only=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            fitting_entrypoint_qualification_remaining=True, new_residual_zero_head_bytes_verified=True, correction_units_guide_RMS=True,
            guide_autograd=False, gradient_vectors_exported=False)
    except BaseException as error:
        result.update(status='failed', error=repr(error), denied_ambient_paths=denied)
        raise
    finally:
        guarded = False
        if model is not None:
            model.close()
        if bank is not None:
            bank.close()
        os.chdir(original_root)
        temporary.cleanup()
        result['elapsed_seconds'] = time.monotonic() - began
        (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
