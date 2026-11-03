"""Export inspectable scalar comparisons; no field arrays or model runtime.

Example: --identifier C106 --identifier C107 --group-by density
Use source metadata/metric definitions when interpreting the resulting CSV.
"""
import argparse
import csv
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

from import_common_generator_measurements import ROOT, DOC, read, binding, require
from portfolio_statistics import distribution


def run(args):
    source = DOC/'reports/pressure-field-statistics-01.json'
    report = read(source)
    spec = report['scalar_case_payload']
    require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Bound scalar payload changed')
    models = {m['fit_item_id']:m for m in report['models'] if not args.identifier
        or m['identifier'] in args.identifier or m['fit_item_id'] in args.identifier}
    require(models, 'No matching pressure model identifier')
    if args.identifier:
        requested = set(args.identifier)
        known = {k for fid,m in models.items() for k in (fid,m['identifier'])}
        require(requested <= known, 'Some requested model identifiers are unknown')
    metadata = report['case_metadata']
    groups = defaultdict(list)
    unavailable = defaultdict(int)

    def group_value(name,case):
        if name == 'density':
            return case['dimensions']['cells_per_metre']
        if name == 'grid':
            return 'x'.join(map(str,case['identity']['counts_xyz']))
        if name == 'force_frequency_hz':
            return case['scene'].get('force',{}).get('frequency_hz','unavailable')
        if name == 'scene':
            return case['identity']['trajectory_id']
        return case['source_population']

    with gzip.open(ROOT/spec['path'],'rt') as stream:
        for line in stream:
            record = json.loads(line)
            fid = record['fit_item_id']
            if fid not in models:
                continue
            case = metadata[record['case_index']]
            if args.density and case['dimensions']['cells_per_metre'] not in args.density:
                continue
            key = (fid,*[group_value(n,case) for n in args.group_by])
            value = record['statistics']
            for part in args.metric.split('.'):
                require(isinstance(value,dict) and part in value, 'Unknown scalar metric path: '+args.metric)
                value = value[part]
            if value is None:
                unavailable[key] += 1
            else:
                require(type(value) in (float,int), 'Requested metric is not a numeric scalar')
                groups[key].append((value,record['case_index']))
    rows = []
    for key in sorted(set(groups)|set(unavailable),key=lambda k:tuple(map(str,k))):
        fid = key[0]
        population = groups[key]
        if population:
            ordered = sorted(v for v,_ in population)
            stats = distribution(ordered)
            stats['min'] = stats.pop('best')
            stats['max'] = stats.pop('worst')
            stats.pop('p95_definition')
            stats['population_std'] = statistics.pstdev(ordered)
            for name,q in (('p90',.9),('p99',.99)):
                position = q*(len(ordered)-1)
                lo = math.floor(position)
                stats[name] = ordered[lo]+(ordered[math.ceil(position)]-ordered[lo])*(position-lo)
        else:
            stats = {name:None for name in ('min','max','mean','median','population_std','p90','p95','p99')}
            stats['samples'] = 0
        row = dict(identifier=models[fid]['identifier'],fit_item_id=fid,task='pressure_request',
            population=report['scope'],metric=args.metric,**dict(zip(args.group_by,key[1:],strict=True)),
            **stats,unavailable_observations=unavailable[key],
            minimum_case=min(population)[1] if population else None,
            maximum_case=max(population)[1] if population else None,
            source=str(source.relative_to(ROOT)))
        rows.append(row)
    require(rows, 'No observed cases match the requested filters')
    stream = args.output.open('x',newline='') if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifier',action='append',default=[])
    parser.add_argument('--metric',default='pressure_mean_free_error.relative_l2')
    parser.add_argument('--density',type=float,action='append',default=[])
    parser.add_argument('--group-by',choices=('density','grid','force_frequency_hz','source_population','scene'),action='append',default=[])
    parser.add_argument('--output',type=Path)
    run(parser.parse_args())
