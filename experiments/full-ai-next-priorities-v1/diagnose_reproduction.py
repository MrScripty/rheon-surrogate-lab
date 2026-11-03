"""Compare the historical no-grad context with inference mode, without updates."""
import json
import os
from study_support import ROOT,BASE,load_checkpoint,historical_cases,equation_error,write,sha

os.sched_setaffinity(0,{23})
state=load_checkpoint('C107')
torch=state.torch
cases=historical_cases(state)
expected=[json.loads(line)for line in (ROOT/'docs/plans/chart-diagnostics/pressure-methods-v4/v4-student50-control/cases.jsonl').read_text().splitlines()]
rows=[]
for index,(identity,sample)in enumerate(cases):
    values={}
    for name,context in [('no_grad',torch.no_grad),('inference_mode',torch.inference_mode)]:
        with context():
            prediction=state.model.predict_pressure(state.network,sample,state.normalization)
            values[name]=equation_error(state,prediction,sample)
    rows.append(dict(**identity,expected=expected[index]['raw_learned']['quality']['relative_residual'],**values))
result=dict(status='diagnostic',rows=rows,source_sha256=sha(__file__),
    worst_absolute_difference={name:max(abs(r[name]-r['expected'])for r in rows)for name in ('no_grad','inference_mode')})
write(BASE/'reproduction-diagnosis-01.json',result)
print(json.dumps(result['worst_absolute_difference']),flush=True)
for r in rows:
    if abs(r['inference_mode']-r['expected'])>max(1e-15,r['expected']*1e-9):print(json.dumps(r),flush=True)
