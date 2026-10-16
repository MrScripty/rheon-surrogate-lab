"""Retain comparable pressure-field drift from completed paired trajectories."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cases=[];table=[];expected=None
    for root in args.studies:
        report=read(root/'report.json');identity=(report['configuration']['scenes'],report['binary_sha256'])
        if expected is None:expected=identity
        if identity!=expected:raise ValueError('same physical scenes and native binary required for drift comparison')
        models={m['name']:m for m in report['models']}
        for row in report['records']:
            if row['model']=='native':continue
            directory=root/f'scene-{row["scene"]:02d}'/f'{row["model"]}-{row["repetition"]}'
            receipt=read(directory/'receipt.json')
            if digest(directory/'receipt.json')!=row['receipt_sha256'] or not receipt['native_process']['reaped']:
                raise ValueError('exact completed/reaped source receipt required')
            scene=receipt['scene'];frames=[read_line for line in (directory/'frames.jsonl').read_text().splitlines()
                if (read_line:=json.loads(line)).get('ai_quality')]
            if len(frames)!=receipt['result']['ai_steps']:raise ValueError('complete AI step population required')
            milestones={scene['warmup_steps']+1,64,128,256,scene['steps']}
            sampled=[]
            for frame in frames:
                if frame['step'] not in milestones:continue
                point={'model':row['model'],'scene':row['scene'],'counts_xyz':'x'.join(map(str,scene['counts'])),
                    'step':frame['step'],'time_s':frame['time_s'],'pressure_relative_l2':frame['relative_pressure_difference'],
                    'ai_relative_residual':frame['ai_quality']['relative_residual'],
                    'reference_relative_residual':frame['reference_quality']['relative_residual'],
                    'ai_actual_divergence':frame['ai_quality']['actual_divergence'],
                    'ai_predicted_divergence':frame['ai_quality']['predicted_divergence'],
                    'qualified':frame['ai_quality']['qualified']}
                sampled.append(point);table.append(point)
            cases.append({'model':row['model'],'weights_sha256':models[row['model']]['weights_sha256'],'scene':scene,
                'source_receipt_sha256':row['receipt_sha256'],'source_report_sha256':digest(root/'report.json'),
                'ai_steps':len(frames),'qualified_ai_steps':sum(f['ai_quality']['qualified'] for f in frames),
                'final_pressure_relative_l2':frames[-1]['relative_pressure_difference'],
                'worst_pressure_relative_l2':max(f['relative_pressure_difference'] for f in frames),
                'worst_ai_actual_divergence':max(f['ai_quality']['actual_divergence'] for f in frames),
                'hard_stop':receipt['result']['hard_stop'],'native_process':receipt['native_process'],'samples':sampled})
    result={'schema':'rheon-multigrid-pressure-drift-summary-v1','cases':cases,
        'scope':'finite development trajectories, common quarter-strength forcing/native controls; pressure-field relative L2 versus native, no asymptotic stability claim'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    args.output.with_suffix('.csv').write_text(stream.getvalue())
    print(json.dumps({'paired_trajectories':len(cases),'sampled_points':len(table),'qualified_ai_steps':sum(r['qualified_ai_steps'] for r in cases)}),flush=True)


if __name__=='__main__':main()
