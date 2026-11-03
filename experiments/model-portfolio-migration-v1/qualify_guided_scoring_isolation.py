"""Reproduce the exact isolation failure and run literal physical score oracles."""
from pathlib import Path
import sys
import tempfile
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from guided_captured_runtime import GuidedCapturedRuntime
from test_macro_readout_fields import Scores


def run():
    output=DOC/'reports/guided-scoring-isolation-qualification-01.json'
    require(not output.exists(), 'Preserve original actual-isolation scorer qualification')
    contract_path=RUN/'guided-component-contract-02.json';contract=read(contract_path)
    require('review_direction_pressure_fits' not in sys.modules, 'Fresh actual lazy-import failure reproduction required')
    test=Scores();test.setUp();item=contract['models'][0];meta=read(ROOT/item['manifest']['path'])
    with tempfile.TemporaryDirectory(prefix='guided-scoring-isolation-',dir=RUN) as job:
        with GuidedCapturedRuntime(item,meta,contract,job):
            try:test.test_half_fields_have_half_relative_error_in_each_physical_metric()
            except ModuleNotFoundError as error:
                require(error.name=='review_direction_pressure_fits', 'Exact original missing scorer module required')
                reproduced=dict(type=type(error).__name__,name=error.name,message=str(error))
            else:raise AssertionError('Original isolation/import failure not reproduced')
    import review_direction_pressure_fits
    rows=[]
    for item in contract['models']:
        meta=read(ROOT/item['manifest']['path'])
        with tempfile.TemporaryDirectory(prefix='guided-scoring-isolation-',dir=RUN) as job:
            with GuidedCapturedRuntime(item,meta,contract,job) as runtime:
                checks=[]
                for name in ('test_half_fields_have_half_relative_error_in_each_physical_metric',
                             'test_pressure_gauge_offset_does_not_hide_its_absolute_diagnostic',
                             'test_zero_target_floors_preserve_absolute_error_and_unknown_raw_relative'):
                    test=Scores();test.setUp();getattr(test,name)();checks.append(name)
                require(not runtime.denied, 'Scoring must use no ambient scientific files')
            rows.append(dict(fit_item_id=item['fit_item_id'],literal_oracles_passed=checks,
                ambient_access_attempts=runtime.denied,temporary_view_removed=not runtime.root.exists()))
    require('torch' not in sys.modules, 'Isolation scorer regression must construct/import no Torch model')
    save(output,dict(status='actual_scorer_isolation_failure_reproduced_and_six_literal_oracles_passed',
        created_unix=time.time(),producer=binding(Path(__file__)),contract=binding(contract_path),
        original_failure=reproduced,models=rows,positive_literal_checks=6,
        correction='Preload the bound independent operator module before captured scientific sys.path isolation. Preserve isolation and all model source/weights.',
        Torch_imported=False,CPU_neural_forwards=0,CUDA_forwards=0,model_dataset_or_project_copies=0,
        native_mutations=0,full_portfolio_complete=False))
    print('Exact lazy-import failure reproduced;all six real-isolation physical-score controls pass without Torch.',flush=True)


if __name__=='__main__':run()
