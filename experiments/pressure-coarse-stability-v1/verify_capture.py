"""Verify original sources, initialization and all full-length sampling arrays.

CPU-only construction of already-consumed inputs: no optimizer, new simulation,
network forward, or acceptance decision. Historical stream hashes are the oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project','artifact-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    sys.path.insert(0,str(args.project.resolve()))
    import torch
    import train
    import analytical
    import direction_feedback
    from gradient_screen import Probe
    from capture_policy import verify_sources
    reference = json.loads((args.project/'gradient-reference.json').read_text())
    witness = json.loads((args.project/'source-witness.json').read_text())
    config = json.loads((args.project/'config.json').read_text())
    verify_sources(args.project, reference)
    for name, identity in witness['sources'].items():
        assert digest(args.project/name) == identity, name
    assert digest(args.project/'config.json') == reference['configuration_sha256']
    project = json.loads((args.project/'project.json').read_text())
    inputs = {}
    for row in project['data']:
        path = args.artifact_root/row['path']
        assert digest(path) == row['sha256'], row['id']
        inputs[row['id']] = path.read_bytes()
    _, parents = train.data.load((inputs['dataset'],inputs['dataset_extra']),json.loads(inputs['splits']))
    train.model.select(config)
    train.model.configure('cpu')
    torch.manual_seed(config['training']['seed'])
    normalization = train.model.fit_normalization(parents,train.data.fit_normalization)
    assert train.sha(train.canonical(normalization)) == witness['normalization_sha256']
    network = train.model.PressureCorrectionNet()
    events = []
    probe = Probe(train,1500,reference,events.append)
    probe.install()
    try:
        probe.attach_network(network)
        train.order(parents,config)
        cache = {(i,j):train.data.sample(parent,j,device='cpu') for i,parent in enumerate(parents)
                 if parent['entry']['split']=='train' for j in range(len(parent['frames']))}
        amplitude = analytical.training_correction_rms(cache.values())
        template = next(iter(cache.values()))
        first = analytical.bank(template,config['analytical_feedback'],5000,amplitude_pa=amplitude)
        second = analytical.supplemental_bank(template,config['supplemental_feedback'],5000,
            first,config['analytical_feedback'],amplitude)
        resident = first[2]['resident_tensor_bytes']+second[2]['resident_tensor_bytes']
        maximum = min(config['analytical_feedback']['cache_max_bytes'],config['supplemental_feedback']['cache_max_bytes'])
        train_ids = [p['entry']['trajectory_id'] for p in parents if p['entry']['split']=='train']
        third = direction_feedback.bank(inputs,config['direction_feedback'],normalization,train_ids,
            5000,'cpu',resident,maximum)
        selected = dict(splits=inputs['splits'],direction_manifest=inputs['current_manifest'],
                        direction_dataset=inputs['current_dataset'])
        direction_feedback.bank(selected,config['current_feedback'],normalization,train_ids,
            5000,'cpu',resident+third[2]['resident_tensor_bytes'],maximum)
        assert set(probe.orders) == set(reference['sampling_order_sha256'])
        # Resolve actual selected identities at the last possible diagnostic step.
        probe.update = 1500
        selected_identities = {name:probe.stream_identity(name,dict(parents=parents,indices=probe.orders['real'][1499]))
                               for name in probe.orders}
    finally:
        probe.close()
    assert not torch.cuda.is_initialized()
    result = dict(format='pressure-coarse-clipping-capture-verification-v1',status='passed',
        elapsed_s=time.perf_counter()-started, all_other_original_numerical_sources_exact=True, allowed_clipping_owner_change_only=True,
        original_configuration_exact=True, original_initial_tensors_exact=True,
        all_full_length_streams_exact=True, events=events, last_cap_sample_identities=selected_identities,
        optimizer_updates=0, network_forwards=0, simulations_advanced=0, cuda_initialized=False,
        reference_sha256=digest(args.project/'gradient-reference.json'),witness_sha256=digest(args.project/'source-witness.json'),
        project_manifest_sha256=digest(args.project/'project.json'),assembly_sha256=digest(args.project/'assembly.json'),
        verified_prefix_updates=1500,
        source_sha256=digest(Path(__file__)),
        scope='Only qualified clipping owner differs. Original5000-length sampling arrays match historical full hashes;1500prefix hashes retained. No optimization or pressure accuracy measurement.')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','elapsed_s','all_full_length_streams_exact','optimizer_updates')}))


if __name__ == '__main__':
    main()
