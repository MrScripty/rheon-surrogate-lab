"""Verify original train-only numerical material without constructing a model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from dependency_config import canonical, decode, load

CHECK = r'''
import sys,json
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import train
from pathlib import Path
p=Path(sys.argv[2]); config=train.experiment(train.decode((p/'config.json').read_bytes()))
train.model.select(config); witness=train.decode((p/'source-witness.json').read_bytes())
assert train.model.ARCHITECTURE==witness['architecture']
manifest=train.decode((p/'inputs/splits.json').read_bytes())
packet,parents=train.data.load(tuple((p/'inputs'/name).read_bytes() for name in ('dataset.npz','dataset_extra.npz')),manifest)
assert parents and all(parent['entry']['split']=='train' for parent in parents)
assert train.sha(train.canonical(packet))==witness['packet_experiment_sha256']
normalization=train.model.fit_normalization(parents,train.data.fit_normalization)
assert train.sha(train.canonical(normalization))==witness['normalization_sha256']
assert len(train.order(parents,config))==1
assert all(case['split']=='train' for case in train.cases(parents))
direction=train.decode((p/'inputs/direction_manifest.json').read_bytes())
assert direction['packet_splits_sha256']==train.sha((p/'inputs/splits.json').read_bytes())
assert train.sha((p/'inputs/direction_manifest.json').read_bytes())==config['direction_feedback']['manifest_sha256']
assert set(direction['train_trajectory_ids'])=={parent['entry']['trajectory_id'] for parent in parents}
print(json.dumps({'status':'passed_train_only_material_contract','parents':len(parents),
 'snapshots':sum(len(parent['frames']) for parent in parents),
 'normalization_sha256':witness['normalization_sha256'],'architecture_matches':True,
 'numeric_input_verified':True,'optimizer_updates':0,'heldout_parents':0}))
'''


def check(selection, project):
    runtime = selection.runtime("cpu")
    manifest = decode(Path(runtime["manifest"]).read_bytes())
    command = [runtime["python"], "-I", "-c", CHECK, manifest["site_packages"], str(Path(project).resolve())]
    started = time.monotonic()
    result = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                            check=True, capture_output=True, text=True, timeout=60)
    return {**decode(result.stdout), "elapsed_seconds": time.monotonic() - started,
            "runtime_manifest_sha256": runtime["manifest_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-config", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(load(args.dependency_config), args.project)
    with args.output.open("xb") as stream:
        stream.write(canonical(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
