"""Reproduce sample identities from recorded seeds/hashes; no model or Torch import."""
import ast
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require


def run():
    path = RUN/'causal-request-sampling-membership-01.json';require(not path.exists(), 'Preserve original sampling replay')
    material_path = RUN/'remaining-component-material-01.json';material = read(material_path)
    rows = []
    for fid in ('fit-a9e199535346c3d9','fit-ccb969376d283836'):
        entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
        source = next(row for row in entry['sources'] if row['original_name'] == 'pressure_request_train.py')
        require(binding(ROOT/source['path'])['sha256'] == source['sha256'], 'Captured sampler source changed')
        tree = ast.parse((ROOT/source['path']).read_bytes())
        sample = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'sample_window')
        body = ast.fix_missing_locations(ast.Module(body=[sample],type_ignores=[]));scope = {}
        # Only the original pure RNG selection function runs, never the trainer.
        exec(compile(body,source['path'],'exec'),scope)
        config = read(ROOT/entry['original_configuration']['path']);history = read(ROOT/entry['outputs']['history.json']['path'])
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        count = config.get('main_updates',config['updates'])
        samples = np.asarray([scope['sample_window'](rng) for _ in range(count)],dtype='<i8')
        require(hashlib.sha256(samples.tobytes()).hexdigest() == history['sample_order_sha256']
                and count == 10000 and samples[:,0].min() >= 0 and samples[:,0].max() < 16
                and samples[:,1].min() == 0 and samples[:,1].max() == 120, 'Original main-update sample sequence differs')
        membership = Counter(tuple(map(int,row)) for row in samples)
        counts = [membership[(parent,step)] for parent in range(16) for step in range(128)]
        rows.append(dict(fit_item_id=fid,main_updates=count,recorded_sample_order_sha256=history['sample_order_sha256'],
            counts_by_case=counts,sampled_distinct_transitions=sum(value > 0 for value in counts),
            unobserved_same_parent_transitions=sum(value == 0 for value in counts),
            outside_sampler_support_transitions=112,source=source,configuration=entry['original_configuration'],
            history=entry['outputs']['history.json']))
    require(rows[0]['counts_by_case'] == rows[1]['counts_by_case'] and 'torch' not in sys.modules, 'Matched pure-metadata sampling replay required')
    save(path,dict(status='both_original_main_q_sampling_sequences_hash_verified',created_unix=time.time(),fits=rows,
        material=binding(material_path),cases_per_fit=2048,eligible_main_sampling_support_cases=1936,
        outside_main_sampling_support_cases=112,independent_holdout=False,
        scope='Replay of original main q-loss selections only;factored head also has a distinct1000update linear calibration. Same-parent unsampled windows remain correlated diagnostic data.',
        Torch_imported=False,CPU_neural_forwards=0,new_parameter_fits=0,new_reference_generation=False,producer=binding(Path(__file__))))
    print('Both10000update sample-order hashes exact;distinctsampled transitions',rows[0]['sampled_distinct_transitions'],flush=True)


if __name__ == '__main__':
    run()
