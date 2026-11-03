"""Compare native JSON decoding on identical original Run documents.

CPU metadata parsing only: excludes snapshot loading, output hashing, SQL,
ProjectStore freshness, GUI latency, model inference and scientific timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from qualify_native_workspace import ROOT


def run(source, report):
    base = ROOT / 'runs/model-portfolio-migration-v1'
    if (report.exists() or not report.is_relative_to(ROOT / 'docs/plans/model-portfolio-migration-v1/reports')
            or not source.is_relative_to(base / 'sources')):
        raise ValueError('Task-owned native source and fresh report required')
    snapshot = base / 'private-history-rehearsal-snapshot-01'
    summary = json.loads((snapshot / 'summary.json').read_bytes())
    with (snapshot / 'metadata.json').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != summary['metadata_sha256']:
            raise ValueError('Original private snapshot changed')
    document = json.loads((snapshot / 'metadata.json').read_bytes())
    table = document['tables']['workspace_runs']
    index_id, index_document = (table['columns'].index(k) for k in ('id','document'))
    selected = [(r[index_id],r[index_document]) for r in sorted(table['rows'], key=lambda r:r[index_id])[:12]]
    del document, table
    api_path = source / 'agent_api.py'
    api_sha = hashlib.sha256(api_path.read_bytes()).hexdigest()
    sys.path.insert(0,str(source))
    import agent_api
    inputs, outputs = hashlib.sha256(), hashlib.sha256()
    measured=[]
    for identity, raw in selected:
        raw_bytes=raw.encode();inputs.update(len(raw_bytes).to_bytes(8,'big'));inputs.update(raw_bytes)
        started=time.process_time()
        value=agent_api.decode(raw)
        elapsed=time.process_time()-started
        normalized=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        outputs.update(len(normalized).to_bytes(8,'big'));outputs.update(normalized)
        measured.append(dict(run_id=identity, input_bytes=len(raw_bytes), decode_CPU_seconds=elapsed))
    if api_sha != hashlib.sha256(api_path.read_bytes()).hexdigest() or 'torch' in sys.modules:
        raise ValueError('Native source changed or model runtime imported during metadata benchmark')
    result=dict(schema='rheon.native-JSON-Run-decoding-benchmark.v1', scope=__doc__.strip(),
                source=str(source.relative_to(ROOT)), source_commit=subprocess.check_output(
                    ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                agent_api_sha256=api_sha, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_snapshot_sha256=summary['metadata_sha256'], records=measured,
                input_sequence_sha256=inputs.hexdigest(), output_sequence_sha256=outputs.hexdigest(),
                total_decode_CPU_seconds=sum(r['decode_CPU_seconds'] for r in measured),
                models_executed=0, SQL_opened=False, source_changed=False)
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('total_decode_CPU_seconds','input_sequence_sha256','output_sequence_sha256')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    run(args.source.resolve(),args.report.resolve())
