"""Describe original saved registrations and exact portable weight coverage.

Reads retained metadata only. Counts of recipes/evidence are structural facts,
not model relevance, fit counts or proof of file availability. No native store,
SQL service, model runtime or captured recipe is executed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from compact_retired_catalog import digest, write
from verify_native_composition import require

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/model-portfolio-migration-v1'
CENSUS=BASE/'discovery/registration-scope-census-01.json'
LOCATIONS=BASE/'discovery/retained-model-locations-01.json'
CUSTODY=ROOT/'docs/plans/model-portfolio-migration-v1/reports/published-composition-custody-01.json'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def classify(project,details,report):
    require(not details.exists() and not report.exists(),'Fresh classification outputs required')
    custody=json.loads(CUSTODY.read_bytes());header=project/'.tuldok/project.json'
    require(digest(header)==custody['composition_manifest_sha256'],'Explicit composed checkpoint required')
    descriptor=json.loads(header.read_bytes())
    declared={r['id'] for r in descriptor['inventory'] if r['family']=='project_artifact'}
    present={p.name for p in (project/'artifacts/sha256').iterdir()}
    weights=json.loads(LOCATIONS.read_bytes())['checkpoints'];known={r['sha256'] for r in weights}
    require(len(known)==524,'Complete known-state locator population required')
    def coverage(value):return 'present' if value in present else 'declared_absent' if value in declared else 'not_declared'
    index=[];catalogs=[];selected=defaultdict(set);variants=defaultdict(set)
    census=json.loads(CENSUS.read_bytes());original_refs=set();outside_refs=set()
    for catalog in census['catalogs']:
        source=ROOT/catalog['source_metadata']
        require(digest(source)==catalog['source_metadata_sha256'],'Original metadata fingerprint differs')
        captured=json.loads(source.read_bytes());table=captured['tables']['workspace_projects']
        outside={r['id']:r for r in catalog['outside_current_native_scope']}
        ids=set();counts=Counter();extensions=Counter();entry_types=Counter()
        for row in table['rows']:
            cells=dict(zip(table['columns'],row));raw=cells['document'];doc=json.loads(raw)
            require(cells['id']==doc['id'] and doc['id'] not in ids,'Original registration identity differs')
            ids.add(doc['id']);manifest=doc['manifest'];reference=doc['manifest_sha256']
            normalized=sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())
            variants[reference].add(normalized)
            is_outside=doc['id'] in outside
            if not is_outside:selected[reference].add(normalized)
            else:
                expected=outside[doc['id']]
                require(doc['name']==expected['name'] and reference==expected['manifest_sha256'],
                        'Outside-scope census identity changed')
            fields={};refs=[]
            for field in ('data','evidence','recipes','extensions'):
                entries=manifest.get(field,[])
                require(isinstance(entries,list),'Unsupported original manifest field '+field)
                fields[field]=len(entries)
                for item in entries:
                    require(isinstance(item,dict),'Unsupported original manifest entry')
                    value=item.get('sha256')
                    if isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value):
                        refs.append(value);original_refs.add(value)
                        if is_outside:outside_refs.add(value)
                    if is_outside:
                        entry_types[field]+=1
                        if isinstance(item.get('path'),str):extensions[Path(item['path']).suffix or '(no suffix)']+=1
            if is_outside:
                counts['registrations']+=1
                counts['with_declared_recipes' if fields['recipes'] else 'without_declared_recipes']+=1
                if doc['name'].startswith('Historical imported evidence '):counts['historical_imported_evidence_named']+=1
                index.append(dict(catalog=catalog['catalog'],id=doc['id'],name=doc['name'],
                    manifest_sha256=reference,normalized_manifest_sha256=normalized,
                    raw_registration_document_sha256=sha(raw.encode()),
                    census_document_sha256=outside[doc['id']]['document_sha256'],
                    fields=fields,direct_sha256_references=dict(Counter(coverage(v) for v in refs)),
                    known_weight_references=sorted(set(refs)&known)))
        require(len(ids)==catalog['total_original_registrations'] and set(outside)<=ids,
                'Original registration population differs')
        catalogs.append(dict(catalog=catalog['catalog'],total_original=len(ids),exact_native=catalog['exact_native_registrations'],
                             outside=dict(counts),outside_entries=dict(entry_types),outside_path_suffixes=dict(extensions)))
        del captured
    for row in index:
        row['same_manifest_as_selected_registration']=row['normalized_manifest_sha256'] in selected.get(row['manifest_sha256'],set())
    weight_rows=[dict(row,portable_status=coverage(row['sha256']),
                     mentioned_by_outside_registration=row['sha256'] in outside_refs) for row in weights]
    require(digest(header)==custody['composition_manifest_sha256'],'Composed checkpoint changed during classification')
    detail=dict(schema='rheon.registration-scope-classification.v1',created_unix=time.time(),
                checkpoint=descriptor['checkpoint'],source_census_sha256=digest(CENSUS),
                source_locator_sha256=digest(LOCATIONS),outside_registrations=index,
                known_checkpoint_states=weight_rows,
                same_manifest_hash_with_distinct_normalized_values={k:sorted(v) for k,v in variants.items() if len(v)>1})
    write(details,detail,True)
    summary=dict(schema=detail['schema'],status='original_registration_structure_and_portable_weight_membership_classified',
        created_unix=time.time(),project=str(project.relative_to(ROOT)),checkpoint=descriptor['checkpoint'],
        catalogs=catalogs,outside_total=len(index),
        outside_same_manifest_as_selected=sum(r['same_manifest_as_selected_registration'] for r in index),
        outside_distinct_manifest_sha256=len({r['manifest_sha256'] for r in index}),
        same_manifest_hash_distinct_values=len(detail['same_manifest_hash_with_distinct_normalized_values']),
        outside_distinct_direct_references=len(outside_refs),outside_reference_coverage=dict(Counter(coverage(v) for v in outside_refs)),
        known_checkpoint_states=len(weights),portable_weight_coverage=dict(Counter(r['portable_status'] for r in weight_rows)),
        known_checkpoint_bytes=sum(r['bytes'] for r in weights),not_portably_present_weight_bytes=sum(r['bytes'] for r in weight_rows if r['portable_status']!='present'),
        details=str(details.relative_to(ROOT)),details_sha256=digest(details),runner_sha256=digest(Path(__file__)),
        source_metadata=[dict(catalog=c['catalog'],path=c['source_metadata'],sha256=c['source_metadata_sha256']) for c in census['catalogs']],
        limitation='Direct top-level entry SHA references only. This does not assess model relevance, path-only files or physical source availability, nor infer missing fits/unique weights. Known checkpoint locations come from the separate prior actual byte audit; no model deserialization or new weight rehash here.',
        payload_files_copied=0,models_executed=0,native_project_modified=False)
    write(report,summary,True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--details',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    classify(args.project.resolve(),args.details.resolve(),args.report.resolve())
