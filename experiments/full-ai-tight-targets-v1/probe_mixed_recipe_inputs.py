"""Exercise the actual recipe decoder and verify every declared static input cap."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runs/dependency-migration-v1/Tuldok'))
import recipe_contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.project / 'project.json').read_text())
    data = {row['id']: row for row in manifest['data']}
    for phase in ('train', 'restore'):
        path = args.project / (phase + '-recipe.json')
        try:
            recipe = recipe_contract.decode_definition(path.read_bytes())
            for name, row in data.items():
                source = ROOT / row['path']
                if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('Actual static input identity changed')
                if source.stat().st_size > recipe['inputs'][name]['max_bytes']:
                    raise ValueError('Actual static input exceeds declared bytes: ' + name)
        except Exception as error:
            print(json.dumps(dict(signal='RED', phase=phase, actual_decoder=True,
                                  error_type=type(error).__name__, error=str(error))), flush=True)
            raise SystemExit(1)
        print(json.dumps(dict(signal='GREEN', phase=phase, actual_decoder=True,
                              all_static_input_sizes_and_hashes_verified=True)), flush=True)
