"""Re-read every bounded current case and bind the complete scientific gate."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import qualify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','bindings','phase','old-cases','binary','manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report={'schema':'rheon-migration-independent-full-cohort-v1','command':sys.argv,
            'source_sha256':qualify.digest(__file__),'qualifier_sha256':qualify.digest(qualify.__file__),
            'bindings_sha256':qualify.digest(args.bindings),'old_cases_sha256':qualify.digest(args.old_cases),
            'binary_sha256':qualify.digest(args.binary),'manifest_sha256':qualify.digest(args.manifest),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'optimizer_updates':0,
            'timing_status':'advisory matched timing remains separate from scientific acceptance'}
    try:
        bindings=qualify.read(args.bindings);reg=bindings['regression'];old=qualify.read(args.old_cases)
        report['fixed_assets_completion_admission']=qualify.admission(args.root,bindings)
        scenes=json.loads(qualify.bound_bytes(args.root,reg['membership_source']))['scenes']
        build=qualify.read(args.binary.parent/'build-receipt.json')
        qualify.require(build['status']=='qualified_build' and build['binary_sha256']==report['binary_sha256'],'qualified current build required')
        qualify.require(all(qualify.digest(path)==value for path,value in build['source_sha256'].items()),'current build input changed')
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/local-pressure-v1'))
        import dependency
        selected=dependency.selection(build['dependency_config'])
        qualify.require(dependency.verify(selected.source_root('rheon'),selected)==build['native_receipt'],'current Rheon source changed')
        manifest=qualify.read(args.manifest)
        for name in ('weights','normalization'):
            filename='weights.npz' if name=='weights' else 'normalization.json'
            item=manifest['files'][filename];expected=reg[name]['sha256']
            qualify.require(item['sha256']==expected and qualify.digest(args.manifest.parent/filename)==expected,'fixed package '+name+' changed')
        workers=[];rows=[];native=[];case_receipts=[];sparse_count=0;active_sources=None
        for worker,indices,cpu in (('worker-a',range(38),1),('worker-b',range(38,74),2)):
            path=args.phase/worker/'receipt.json';receipt=qualify.read(path)
            qualify.require(receipt['status']=='passed' and not receipt['failures'] and [r['scene'] for r in receipt['records']]==list(indices),'complete successful worker required')
            qualify.require(receipt['cpu']==cpu and receipt['threads']==1 and receipt['workers']==1,'worker resource policy changed')
            qualify.require(receipt['binary_sha256']==report['binary_sha256'] and receipt['manifest_sha256']==report['manifest_sha256'],'worker binary/package mismatch')
            qualify.require(all(qualify.digest(path)==value for path,value in receipt['sources'].items()),'worker source changed before aggregate')
            workers.append({'path':str(path.resolve()),'sha256':qualify.digest(path),
                            'elapsed_seconds':receipt['elapsed_seconds'],'peak_child_rss_bytes':receipt['peak_child_rss_bytes'],
                            'minimum_available_memory_bytes':min(r['available_memory_before_bytes'] for r in receipt['records']),
                            'sources':receipt['sources'],'cpu':cpu})
            for index,record in zip(indices,receipt['records']):
                directory=Path(record['output']);measured=qualify.read(directory/'receipt.json')
                qualify.require(measured['status']=='passed' and measured['binary_sha256']==report['binary_sha256']
                                and measured['manifest_sha256']==report['manifest_sha256']
                                and measured['build_receipt_sha256']==qualify.digest(args.binary.parent/'build-receipt.json')
                                and measured['native_source']==build['native_receipt'],'case source/binary/package witness mismatch')
                if active_sources is None:active_sources=measured['active_sources']
                qualify.require(measured['active_sources']==active_sources and all(qualify.digest(path)==value for path,value in active_sources.items()),'case implementation changed')
                qualify.require(measured['cpu_affinity']==[cpu] and measured['torch_threads']==1 and measured['cuda_initialized'] is False
                                and measured['optimizer_updates']==0,'case execution policy mismatch')
                scene=scenes[index];metrics=qualify.case_metrics(directory,scene,reg['cases'][index],reg['stopping_values'],report['binary_sha256'])
                qualify.require(metrics==measured['metrics'],'re-read case metrics differ')
                control={'scene':index,**qualify.native_metrics(directory/'native-control',scene,report['binary_sha256'])}
                qualify.require(control==record['native_control'],'re-read native control differs')
                qualify.accept_case(reg['cases'][index],old[index],metrics)
                for capture in measured['captures']:
                    qualify.require(qualify.digest(capture['path'])==capture['sha256'],'sparse capture changed')
                sparse_count+=len(measured['sparse']);rows.append({**metrics,'output':str(directory)})
                native.append(control);case_receipts.append({'scene':index,'path':str(directory/'receipt.json'),'sha256':qualify.digest(directory/'receipt.json')})
        result=qualify.compare_cohorts(bindings,old,rows,native)
        for name,value in (('new-cases.json',rows),('new-native-controls.json',native)):
            with (args.output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
        raw_bytes=sum(path.stat().st_size for path in args.phase.rglob('*') if path.is_file())
        qualify.require(raw_bytes<1024**3,'1 GiB raw phase budget exceeded')
        report.update(status='passed',result=result,workers=workers,case_receipts=case_receipts,
                      ai_steps=sum(len(row['calls']) for row in rows),sparse_same_rhs_samples=sparse_count,
                      all_call_sequences_exact_to_reproduced_old=all(a['calls']==b['calls'] for a,b in zip(old,rows)),
                      all_pressure_maxima_exact_to_reproduced_old=all(a['worst_paired_pressure_relative_l2']==b['worst_paired_pressure_relative_l2'] for a,b in zip(old,rows)),
                      native_controls=74,raw_phase_bytes=raw_bytes,active_sources=active_sources,
                      native_source=build['native_receipt'],build_receipt_sha256=qualify.digest(args.binary.parent/'build-receipt.json'),
                      new_cases_sha256=qualify.digest(args.output/'new-cases.json'),native_controls_sha256=qualify.digest(args.output/'new-native-controls.json'))
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (args.output/'receipt.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({key:report[key] for key in ('status','ai_steps','native_controls','sparse_same_rhs_samples','elapsed_seconds')}))


if __name__=='__main__':main()
