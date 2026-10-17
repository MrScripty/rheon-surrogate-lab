"""Package a verified fresh-CPU Study result using the current adapter owner."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dependency-migration-v1'))
from dependency_config import decode, digest, require
from package_model_adapter import package


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','study','runtime','runtime-owner','iteration','output','archive'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();project=a.project.resolve(strict=True);study=a.study.resolve(strict=True)
    receipt=decode((study/'receipt.json').read_bytes());require(receipt['status']=='completed' and all(receipt['artifact_parity'].values()),'qualified completed Study required')
    artifacts=study/'restore';cpu=decode((artifacts/'receipt.json').read_bytes());require(cpu['mode']=='restore' and not cpu['execution']['cuda_initialized'] and cpu['optimizer_updates']==0,'fresh CPU restore required')
    require(digest(artifacts/'weights.npz')==receipt['weights_sha256'],'actual Study weights required')
    config=decode((project/'config.json').read_bytes());witness=decode((project/'source-witness.json').read_bytes())
    require(cpu['closure']['source_witness_sha256']==digest(project/'source-witness.json'),'exact captured trainer source required')
    for name,sha in witness['sources'].items():require(digest(project/name)==sha,'captured source changed')
    history=decode((artifacts/'history.json').read_bytes());inventory=history['checkpoints'][f"checkpoint-{config['training']['updates']:03d}.npz"]['inventory']
    originals={**{'source/'+p.name:p for p in project.iterdir() if p.is_file()},
        **{'artifacts/'+name:artifacts/name for name in ('weights.npz','normalization.json','receipt.json','history.json')},'study-receipt.json':study/'receipt.json'}
    result=package('physical_residual',project,artifacts,inventory,a.runtime.resolve(strict=True),a.output.resolve(),a.archive.resolve(),originals,
        dict(study_receipt_sha256=digest(study/'receipt.json'),training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id'],band_feedback=receipt['band_feedback']),
        iteration=a.iteration.resolve(strict=True),runtime_owner=a.runtime_owner.resolve(strict=True),package_id=config['id'],
        spatial_coarse='spatial_coarse_correction' in cpu['architecture'],
        package_version='pressure-band-feedback-v1',role='prospective trained candidate; accuracy unqualified; no promotion')
    print(json.dumps(result))

if __name__=='__main__':main()
