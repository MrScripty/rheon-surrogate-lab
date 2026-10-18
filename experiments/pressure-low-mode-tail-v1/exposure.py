"""Count actual supplemental spectral exposure from the retained sampler order."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def exposure(project, study, output):
    read = lambda path:json.loads(path.read_text())
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    witness = read(project/'source-witness.json')
    if sha(project/'analytical.py') != witness['sources']['analytical.py']:
        raise ValueError('Captured generator changed')
    receipt = read(study/'train/receipt.json')
    if sha(study/'train/history.json') != receipt['outputs']['history.json']['sha256']:
        raise ValueError('Retained training history changed')
    history = read(study/'train/history.json')
    bank = history['supplemental_feedback']
    sys.path.insert(0, str(project))
    import analytical
    import torch
    grids = bank['grids']
    size = len(grids[0]['bank']['samples'])
    order, _ = analytical._balanced_order(len(grids), size, receipt['optimizer_updates'], bank['seed'])
    if hashlib.sha256(order.tobytes()).hexdigest() != bank['sampling_order_sha256']:
        raise ValueError('Exact retained sampling order differs')
    counts = [dict(grid=g['grid'], total=0, fundamental_axis=0, pure_low_block=0, any_low_component=0) for g in grids]
    for raw in order:
        grid, slot = divmod(int(raw), size)
        components = grids[grid]['bank']['samples'][slot]['components_zyx']
        row = counts[grid]
        low = [all(k <= 3 for k in mode) for mode, _ in components]
        row['total'] += 1
        row['fundamental_axis'] += len(components)==1 and sum(components[0][0])==1
        row['pure_low_block'] += all(low)
        row['any_low_component'] += any(low)
    totals = {key:sum(r[key] for r in counts) for key in ['total','fundamental_axis','pure_low_block','any_low_component']}
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only scalar analysis')
    result = dict(status='passed', totals=totals, grids=counts,
        definition='Actual 1500 supplemental draws. Fundamental means one single-axis index-1 mode. Pure low block means every component has all indices <=3. Primary/real/direction/current losses are outside these counts.',
        interpretation='Grid balancing is not spectral balancing. Exposure alone does not prove the cause of regression; test single-mode fit before choosing broader training or an architecture change.',
        sampling_order_sha256=bank['sampling_order_sha256'], history_sha256=sha(study/'train/history.json'),
        generator_sha256=sha(project/'analytical.py'), source_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(totals,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project','study','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    exposure(args.project.resolve(),args.study.resolve(),args.output.resolve())
