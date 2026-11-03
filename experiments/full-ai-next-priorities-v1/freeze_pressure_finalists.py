"""Freeze all three learned-head seeds before independent pressure capture."""
import time
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


if __name__ == '__main__':
    output = DOCS / 'pressure-finalist-selection.json'
    if output.exists():
        raise ValueError('Finalist selection is immutable')
    batch = BASE / 'routed-qualification-batch-01/execution.json'
    if read(batch)['status'] != 'completed':
        raise ValueError('All routed components must qualify before freezing')
    models = []
    entry_path = ROOT / 'runs/pressure-methods-v4/entries/v4-student50-control.json'
    models.append(dict(id='C107', entry_path=str(entry_path.relative_to(ROOT)), entry_sha256=sha(entry_path), role='baseline'))
    ranking = []
    for suffix in ('', '-s2', '-s3'):
        identifier = 'a-head-calibration' + suffix + '-routed'
        qualification = BASE / (identifier + '-qualification-01.json')
        evidence = read(qualification)
        if evidence['status'] != 'passed' or evidence['tested_requests'] != 196 or evidence['cuda_initialized']:
            raise ValueError('Actual fresh CPU equivalence qualification required')
        historical = [row for row in evidence['rows'] if row['split'] == 'consumed_historical_development']
        worst = max(row['equation_relative_residual'] for row in historical)
        if len(historical) != 128 or worst > 7e-4:
            raise ValueError('Every seed must pass the historical accuracy criterion')
        entry_path = BASE / 'entries' / (identifier + '.json')
        ranking.append((worst, identifier))
        models.append(dict(id=identifier, entry_path=str(entry_path.relative_to(ROOT)), entry_sha256=sha(entry_path),
                           role='replicated_finalist', historical_worst_equation_error=worst,
                           qualification_path=str(qualification.relative_to(ROOT)), qualification_sha256=sha(qualification)))
    sources = ['fresh_pressure_data.py', 'evaluate_fresh_pressure.py', 'benchmark_fresh_pressure.py',
               'capture_pressure_fresh.py', 'freeze_pressure_finalists.py', 'routed_inference.py', 'study_support.py']
    write(output, dict(status='frozen_before_reserved_capture', frozen_unix=time.time(), models=models,
          primary_model=min(ranking)[1], selection_rule='Lowest historical worst equation error among qualified seed replications',
          scene_reservation_sha256=sha(DOCS/'pressure-scene-reservation.json'),
          execution='Original eager CPU float32 fine network and learned head; all stages and reconstruction included',
          compiled_execution_decision='Retain as diagnostic: qualified but timing gain inconsistent across seeds; eager larger-grid component equivalence retained',
          compiled_timing_sha256=sha(BASE/'pressure-compiled-timing-01/summary.json'),
          qualification_batch_sha256=sha(batch), candidate_numerical_pressure_solve=False,
          candidate_corrective_projection=False, reserved_temporal_scenes_consumed=False,
          source_sha256={name:sha(HERE/name) for name in sources},
          prepared_request_evaluation='All1024 requests:768small and25632cubed; populations reported separately',
          cpu_timing='Five balanced serial repeats, every scene at steps1,16,32,64; one thread CPU23 batch1',
          native_evaluation='All16scenes,64steps requested, original acceptance gates, stop on rejection, no fallback',
          interpretation='Reserved outcomes are reporting evidence; no refitting or seed reselection based on them'))
    print('Frozen three replicated finalists and C107 baseline before fresh capture', min(ranking)[1], flush=True)
