"""Give unpublished plot subjects readable IDs without changing scientific data.

A compact inverse description reconstructs every original packet byte. After
independent qualification, superseded unpublished packet files can be retired;
original raw benchmark files and historical instrumented tables stay intact.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from qualify_common_generator_measurements import ROOT, RUN, DOC, binding, read, require
from prepare_portfolio_measurements import source

OLD = RUN/'matched-generator-GPU-measurements-01'
NEW = RUN/'matched-generator-GPU-measurements-02'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'


def native():
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable,encode,digest
    return MeasurementTable,encode,digest


def rewrite(packet,spec,reverse=False):
    """Only presentation IDs, labels and their exact JSON references change."""
    _,encode,digest = native()
    packet = copy.deepcopy(packet)
    mapping = spec['aliases']
    if reverse:
        mapping = {new:old for old,new in mapping.items()}
    sources = {s['id']:s for s in packet['sources']}
    context = json.loads(base64.b64decode(sources['context']['data_base64']))
    evidence = json.loads(base64.b64decode(sources['reduction']['data_base64']))
    title = spec['old_title'] if reverse else spec['new_title']
    packet['title'] = packet['scope']['label'] = context['scope']['label'] = title
    for subject in packet['subjects']:
        original_id = mapping[subject['id']] if reverse else subject['id']
        subject['id'] = mapping[subject['id']]
        subject['label'] = spec['old_labels'][original_id] if reverse else spec['new_labels'][original_id]
        subject['reference']['pointer'] = '/models/'+subject['id']
    evidence['models'] = {mapping[key]:value for key,value in evidence['models'].items()}
    for row in evidence['rows']:
        row['subject_id'] = mapping[row['subject_id']]
    for observation in packet['observations']:
        observation['subject_id'] = mapping[observation['subject_id']]
    updated = []
    for item in packet['sources']:
        if item['id'] in ('context','reduction'):
            value = context if item['id'] == 'context' else evidence
            updated.append(source(item['id'],item['label'],encode(value)))
        else:
            item = dict(item)
            for old,new in mapping.items():
                if item['id'].startswith(old+'-raw-'):
                    item['id'] = new+item['id'][len(old):]
                    break
            updated.append(item)
    packet['sources'] = updated
    packet['scope']['definition']['sha256'] = digest(context['scope'])
    return packet


def prepare():
    require(not NEW.exists(),'Preserve original readable packet preparation')
    Table,encode,digest = native()
    old_prep = read(OLD/'preparation.json')
    old_proof = DOC/'reports/matched-generator-GPU-measurements-qualification-01.json'
    proof = read(old_proof)
    require(proof['measured_scalars'] == 24960 and len(proof['queries']) == 600,'Complete prior scientific qualification required')
    staged,specs,parts = [],[],[]
    all_aliases = set()
    for part in old_prep['packets']:
        raw = (OLD/part['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == part['sha256'],'Original unpublished packet changed')
        packet = json.loads(raw)
        evidence = json.loads(base64.b64decode(next(s for s in packet['sources'] if s['id'] == 'reduction')['data_base64']))
        aliases,labels = {},{}
        for subject in packet['subjects']:
            identity = subject['id']
            model = evidence['models'][identity]
            if model['origin_kind'] == 'original_fit':
                original = model['original_model']
                alias = original['identifier']
                if alias == 'b1':
                    config = read(ROOT/original['configuration']['path'])
                    require(config['schema'] in ('rheon-temporal-fit-v1','rheon-temporal-fit-v2'),'Exact b1 schema required')
                    alias += '-v1' if config['schema'].endswith('-v1') else '-v2'
            else:
                alias = f'F64-N1-K{part["K"]}'
            require(alias not in all_aliases,'Readable plot ID collision')
            all_aliases.add(alias)
            aliases[identity] = alias
            labels[identity] = alias+f' · GPU · full K{part["K"]} block'
        spec = dict(K=part['K'],path=part['path'],old_sha256=part['sha256'],aliases=aliases,
            old_labels={s['id']:s['label'] for s in packet['subjects']},new_labels=labels,
            old_title=packet['title'],new_title=f'Matched GPU generators and F64 baseline · K{part["K"]} · same 64 cases')
        revised = rewrite(packet,spec)
        require(encode(rewrite(revised,spec,reverse=True)) == raw,'Original packet must reconstruct byte for byte')
        new_raw = encode(revised)
        table = Table(new_raw)
        specs.append(spec)
        staged.append((part['path'],new_raw))
        parts.append(dict(part,sha256=table.sha256,bytes=len(new_raw),subjects=[s['id'] for s in revised['subjects']]))
    require(len(all_aliases) == 39,'All 39 readable identities required')
    declaration = dict(schema_version=1,id=digest(parts)[:32],name='35 generator fits and 4 matched F64 executions · GPU',
        data=[dict(id=f'K{p["K"]}-complete-GPU',label=p['scope'],path=str((NEW/p['path']).relative_to(ROOT)),sha256=p['sha256']) for p in parts],
        recipes=[],evidence=[],requirements=[])
    from projects import manifest
    manifest(declaration)
    NEW.mkdir()
    for name,raw in staged:
        (NEW/name).write_bytes(raw)
    (NEW/'source-project.json').write_bytes(encode(declaration))
    (NEW/'presentation-inverse.json').write_bytes(encode(dict(original_preparation=binding(OLD/'preparation.json'),packets=specs)))
    result = dict(old_prep,status='readable_matched_GPU_tables_prepared_not_adopted',created_unix=time.time(),packets=parts,
        producer=binding(Path(__file__)),prior_preparation=binding(OLD/'preparation.json'),prior_qualification=binding(old_proof),
        source_manifest=binding(NEW/'source-project.json'),presentation_inverse=binding(NEW/'presentation-inverse.json'),
        original_packet_bytes_reconstructible=True,superseded_packet_files_retired=False)
    (NEW/'preparation.json').write_bytes(encode(result))
    print('All 39 plot IDs readable; b1-v1/b1-v2 remain distinct; all four old packets reconstruct exactly.')


def retire():
    Table,encode,digest = native()
    report_path = DOC/'reports/superseded-matched-packet-retirement-01.json'
    require(not report_path.exists(),'Preserve original packet retirement')
    proof_path = DOC/'reports/readable-matched-generator-GPU-measurements-qualification-01.json'
    proof = read(proof_path)
    require(proof['all24960_scientific_observations_exact'] and proof['raw_process_blobs_exact'] == 78
            and proof['native_filter_queries'] == 600,'Independent readable packet qualification required')
    specs = read(NEW/'presentation-inverse.json')['packets']
    retired = []
    for spec in specs:
        path = OLD/spec['path']
        reconstructed = encode(rewrite(read(NEW/spec['path']),spec,reverse=True))
        require(path.is_file() and not path.is_symlink() and hashlib.sha256(reconstructed).hexdigest() == spec['old_sha256']
                and path.read_bytes() == reconstructed,'Only exactly reconstructible unpublished packets may retire')
        retired.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=spec['old_sha256']))
    for entry in retired:
        (ROOT/entry['path']).unlink()
    report = dict(status='four_superseded_unpublished_packet_files_retired_exact_inverse_retained',created_unix=time.time(),
        retired=retired,reclaimed_logical_bytes=sum(e['bytes'] for e in retired),
        inverse=binding(NEW/'presentation-inverse.json'),proof=binding(proof_path),producer=binding(Path(__file__)),
        original_raw_benchmarks_changed=False,historical_instrumented_tables_changed=False,
        unique_weights_deleted=0,native_mutations=0,full_portfolio_complete=False)
    report_path.write_bytes(encode(report))
    print('Retired',report['reclaimed_logical_bytes'],'superseded metadata bytes; no weights/data removed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','retire'))
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else retire()
