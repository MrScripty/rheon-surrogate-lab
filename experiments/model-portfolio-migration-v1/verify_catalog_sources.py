"""Verify retained source blobs with the existing owner's bounded file reader.

The material references come from the authenticated catalog capture. No Dataset
or Metadata instance, SQL connection, model import or inference is constructed.
The reader checks file custody and content hashes; it does not execute sources.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / 'runs/dependency-migration-v1/Tuldok-large-input-v3'
CATALOG = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres')
CATALOG_ID = 'a7da4a8de53541a2a0e2aad08005bb71'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(observation, output, summary_path):
    if output.exists() or summary_path.exists():
        raise ValueError('Fresh source verification outputs required')
    if json.loads((CATALOG / '.tuldok.json').read_text())['dataset_id'] != CATALOG_ID:
        raise ValueError('Selected catalog differs')
    captured = json.loads(observation.read_text())
    sys.path.insert(0, str(SERVICE))
    import model_definitions
    models, failures = [], []
    for record in captured['captured_metadata']['model_materials']:
        ref, material = record['ref'], record['material']
        if ref['catalog_id'] != CATALOG_ID or material['catalog_id'] != CATALOG_ID:
            raise ValueError('Foreign model material reference')
        sources, policies = [], {}
        for source in material['source_closure']:
            item = source['blob']
            try:
                model_definitions.read_owned_blob(CATALOG / 'model-materials', ref['sha256'],
                                                  item, model_definitions.MAX_SOURCE_BYTES)
                sources.append(dict(import_path=source['import_path'], blob=item,
                                    retained_path=str(CATALOG / 'model-materials' / ref['sha256'] / item['name']),
                                    status='verified_owned_source_blob'))
            except model_definitions.WorkbenchError as error:
                failures.append(dict(model=ref, import_path=source['import_path'], code=error.code))
        for key in ('specification', 'preprocessing_policy', 'output_policy'):
            try:
                raw = model_definitions.read_owned_blob(CATALOG / 'model-materials', ref['sha256'],
                                                        material[key], model_definitions.MAX_SOURCE_BYTES)
                policies[key] = json.loads(raw)
            except (model_definitions.WorkbenchError, ValueError) as error:
                failures.append(dict(model=ref, policy=key, error_type=type(error).__name__))
        models.append(dict(ref=ref, entrypoint=material['entrypoint'], sources=sources, policies=policies))
    result = dict(schema='rheon.retained-model-source-verification.v1', created_unix=time.time(),
                  status='verified' if not failures else 'partial_with_failures', models=models,
                  failures=failures, catalog_id=CATALOG_ID,
                  observation=dict(path=str(observation), sha256=sha(observation)),
                  reader=dict(path=str(SERVICE / 'model_definitions.py'), sha256=sha(SERVICE / 'model_definitions.py'),
                              symbol='read_owned_blob'), source_sha256=sha(Path(__file__)),
                  database_opened=False, payloads_executed=False, files_copied=0,
                  limitations=['Retained numerical data, runtimes, weights and reference binaries require their own verification.',
                               'Verified source bytes do not imply a model was scientifically successful.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    summary = dict(schema=result['schema'], status=result['status'], output=str(output), output_sha256=sha(output),
                   models=len(models), source_blobs=sum(len(m['sources']) for m in models),
                   policies=sum(len(m['policies']) for m in models), failures=failures,
                   database_opened=False, payloads_executed=False, files_copied=0)
    with summary_path.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-observation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    verify(args.catalog_observation, args.output, args.summary)
