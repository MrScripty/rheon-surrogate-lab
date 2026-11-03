"""Extract useful spatial scalar diagnostics before retiring prediction fields."""
import hashlib
import io
from pathlib import Path
import sys
import time
import zipfile

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import observed_file
from macro_readout_fields import FIELDS, digest, load_fields
from portfolio_statistics import distribution
from spatial_field_statistics import measure
from guided_captured_runtime import GuidedCapturedRuntime


def run():
    import numpy as np
    output=DOC/'reports/recent-component-spatial-statistics-01.json'
    require(not output.exists(), 'Preserve original spatial scalar extraction')
    results=[];sources=[]
    def capture(model,spec,items,truth_for,scope):
        observed_file(ROOT,spec['path'],spec);rows=[]
        with np.load(ROOT/spec['path'],allow_pickle=False) as archive:
            for index,lead,row,prefix in items:
                identity=dict(row['identity']);truth,counts=truth_for(index,lead)
                predicted={n:archive[prefix+n] for n in FIELDS}
                require({n:digest(v) for n,v in predicted.items()}==row['output_hashes'], 'Accepted physical output hashes must match retained arrays')
                h=1/identity['cells_per_metre'];values=measure(predicted,truth,counts,[h,h,h])
                for name in ('pressure','tracer'):
                    actual=values[name+'_absolute_rms_error'];expected=row['metrics']['absolute_rms'][name]
                    require(abs(actual-expected)<=1e-12+1e-10*abs(expected), 'New spatial score must preserve accepted RMS error')
                rows.append(dict(identity=identity,values=values))
        names=set().union(*(r['values'] for r in rows));summary={}
        for name in names:
            summary[name]=distribution(r['values'][name] for r in rows)
            # Physical values and signed differences have extrema,not an
            # automatic better/worse ordering.
            summary[name]['minimum']=summary[name].pop('best');summary[name]['maximum']=summary[name].pop('worst')
        results.append(dict(fit_item_id=model['fit_item_id'],scope=scope,spatial_outputs=len(rows),
            source_prediction_identity=spec,summary=summary,rows=rows))
    readout_path=DOC/'reports/macro-readout-field-GPU-accuracy-01.json';report=read(readout_path);sources.append(binding(readout_path))
    contract=read(ROOT/report['contract']['path'])
    for model in report['models']:
        items=[(r['case_index'],1,r,f"case{r['case_index']:03d}_") for r in model['rows']]
        def truth_for(index,lead):
            case=contract['cases'][index]
            return load_fields(case['target_state']),case['counts_xyz']
        capture(model,model['prediction_archive'],items,truth_for,'Complete58active-window field population for this readout;consumed fitting scenes')
    correction_path=DOC/'reports/internal-residual-GPU-accuracy-01.json';report=read(correction_path);sources.append(binding(correction_path))
    contract=read(ROOT/report['contract']['path']);manifest=read(ROOT/contract['manifest']['path'])
    with zipfile.ZipFile(ROOT/contract['archive']['path']) as corpus:
        for model in report['models']:
            table={r['case_index']:r for r in model['rows']}
            items=[(i,1,table[i],f'case{i:03d}_') for i in model['retained_case_indices']]
            def truth_for(index,lead):
                spec=manifest['parts'][index]
                raw=corpus.read(spec['member']);require(hashlib.sha256(raw).hexdigest()==spec['sha256'], 'Original target member changed')
                with np.load(io.BytesIO(raw),allow_pickle=False) as stored:
                    truth={n:stored['target_'+n].reshape(-1).copy() for n in FIELDS}
                return truth,spec['counts_xyz']
            capture(model,model['worst_field_capture'],items,truth_for,'Selected original metric-worst request outputs only;not a representative/full783request spatial population')
    guided_path=DOC/'reports/guided-GPU-accuracy-01.json';report=read(guided_path);sources.append(binding(guided_path))
    contract=read(ROOT/report['contract']['path'])
    for model in report['models']:
        item=next(m for m in contract['models'] if m['fit_item_id']==model['fit_item_id']);meta=read(ROOT/item['manifest']['path'])
        table={(r['case_index'],r['lead']):r for r in model['rows']}
        items=[(x['case_index'],x['lead'],table[x['case_index'],x['lead']],f"case{x['case_index']:04d}_lead{x['lead']}_") for x in model['retained_outputs']]
        with GuidedCapturedRuntime(item,meta,contract,output,extra_read_paths=[model['worst_field_capture']['path']]) as runtime:
            from captured_long_training_bank_v1 import CapturedLongTrainingBank
            bank=CapturedLongTrainingBank(ROOT/contract['data_archive']['path'],{n:contract['data_archive'][n] for n in ('bytes','sha256')})
            try:
                def truth_for(index,lead):
                    case=contract['cases'][index]
                    return bank.load(case['identifier'],case['target_fine_steps'][lead-1]),bank.manifest['masters'][case['identifier']]['scene']['counts']
                capture(model,model['worst_field_capture'],items,truth_for,'Selected original metric-worst physical outputs only;not a representative/full7200output spatial population')
                imported=runtime.imported_sources()
            finally:bank.close()
        for name in imported:sys.modules.pop(name,None)
    require('torch' not in sys.modules and len(results)==8, 'No model/Torch runtime or inference belongs to scalar extraction')
    save(output,dict(status='available_recent_fields_reduced_to_spatial_scalars_no_fields_saved',created_unix=time.time(),
        producer=binding(Path(__file__)),spatial_helper=binding(Path(__file__).with_name('spatial_field_statistics.py')),
        source_reports=sources,models=results,total_spatial_outputs=sum(m['spatial_outputs'] for m in results),
        definitions=dict(pressure_error='Mean-free cell error;gauge invariant',velocity_error='Separate original staggered face components',
            divergence='diff_x(vx)/hx+diff_y(vy)/hy+diff_z(vz)/hz;units1/s;reference divergence retained',
            boundary='Only outer normal staggered velocities;sealed-domain diagnostic',
            tracer_integral='sum(tracer)*hx*hy*hz;compared with reference,not assumed conserved under arbitrary sources or numerical transport',
            extrema='Spatial scalar distributions use minimum/maximum;state values and signed errors have no automatic better/worse direction'),
        coverage='Two readouts complete58windows each;other six fits only existing metric-worst capture samples. Missing whole-population spatial diagnostics are explicit future in-memory measurements.',
        prediction_arrays_saved=False,CPU_neural_forwards=0,new_model_forwards=0,native_mutations=0,
        full_portfolio_complete=False))
    print('Spatial scalar diagnostics retained for',sum(m['spatial_outputs'] for m in results),'available outputs;no field files saved.',flush=True)


if __name__=='__main__':run()
