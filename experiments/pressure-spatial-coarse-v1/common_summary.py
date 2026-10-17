"""Compare fixed-call errors on exactly shared train-only RHS pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(paths):
    sources, models = {}, {}
    pair_hashes = preparation = None
    for name,path in paths.items():
        value = json.loads(path.read_text())
        if (value['status'] != 'passed' or not value['all_workers_reaped'] or
                value['network_calls'] != 1152 or len(value['cases']) != 144):
            raise ValueError('complete drained 144-pair diagnostic required')
        current = [row['pair_sha256'] for row in value['cases']]
        if pair_hashes is None:
            pair_hashes, preparation = current,value['preparation_sha256']
        elif current != pair_hashes or value['preparation_sha256'] != preparation:
            raise ValueError('different common RHS pairs or preparation')
        sources[name] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'weights_sha256':value['weights_sha256'],'elapsed_s':value['elapsed_s'],
                         'workers':value['workers'],'network_calls':value['network_calls']}
        models[name] = {}
        for stage in ('early','late'):
            rows = [row for case in value['cases'] for row in case['rows'] if row['stage'] == stage]
            if (len(rows) != 144 or max(row['label_consistency_relative_l2'] for row in rows) >= 1e-9 or
                    max(row['independent_operator_relative_defect'] for row in rows) >= 1e-9):
                raise ValueError('independent common-label/operator check failed')
            keys = ('one_call_mean_free_pressure_relative_l2','two_call_mean_free_pressure_relative_l2',
                    'two_call_relative_residual','native_vs_rescaled_prediction_relative_l2')
            stats = {key:{'median':statistics.median(row[key] for row in rows),
                          'maximum':max(row[key] for row in rows)} for key in keys}
            bins = {}
            for row in rows:
                for band in row['frequency_amplitude']:
                    key = band['band']+':'+','.join(map(str,band['coefficient_over_full_l2_interval']))
                    item = bins.setdefault(key,{'band':band['band'],
                        'coefficient_over_full_l2_interval':band['coefficient_over_full_l2_interval'],
                        'modes':0,'signal_energy_pa2':0.,'error_energy_pa2':0.})
                    for field in ('modes','signal_energy_pa2','error_energy_pa2'):
                        item[field] += band[field]
            for item in bins.values():
                item['error_energy_over_signal_energy'] = (item['error_energy_pa2']/item['signal_energy_pa2']
                                                          if item['signal_energy_pa2'] else None)
            models[name][stage] = {'states':len(rows),'statistics':stats,'frequency_amplitude_bins':list(bins.values())}
    return {'format':'pressure-spatial-coarse-common-rhs-summary-v1','status':'passed',
        'sources':sources,'shared_pairs':144,'same_pair_hashes_and_preparation':True,'models':models,
        'scope':'Exactly matched train-only pressure RHS/labels; two native-amplitude and two rescaled network calls per stage/state/model. Spatial cosine bins are grid/case-weighted energies, not temporal Hz or causal noise-floor evidence. Concurrent clocks are not inference latency.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',required=True,help='LABEL=completed common-RHS receipt')
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    paths = {}
    for item in a.model:
        name,separator,path = item.partition('=')
        if not separator or not name or name in paths:
            raise ValueError('unique model label required')
        paths[name] = Path(path).resolve(strict=True)
    result = summarize(paths)
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':'passed','models':list(paths),'pairs':144}))


if __name__ == '__main__':
    main()
