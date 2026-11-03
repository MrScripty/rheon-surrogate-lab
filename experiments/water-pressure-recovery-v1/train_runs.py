"""Serial managed CUDA smoke and fits; Tuldok retains Run lifecycle authority."""
import argparse
from pathlib import Path
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import command, read, write, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    args = parser.parse_args()
    root = LAB/'runs/water-pressure-recovery-v1'
    status = root/'training-execution.json'
    assert not status.exists(), 'Reconcile retained Runs before another attempt'
    record = {'state':'running','started_at_utc':utc(),'commands':[],
              'controller_source_sha256':__import__('hashlib').sha256(args.controller.read_bytes()).hexdigest()}
    started = time.perf_counter()
    try:
        for name in ('smoke','gentle-010','gentle-025'):
            record['trial'] = name
            write(status, record)
            record['commands'].append(command([sys.executable,args.controller,'--root',root,
                                               '--trial',name,'--phase','full'],root,'managed-'+name))
            artifacts = root/name/'managed-full'
            assert read(artifacts/'receipt.json')['status']=='completed_and_drained'
            trained = read(artifacts/'stage-00/artifacts/receipt.json')
            restored = read(artifacts/'stage-01/artifacts/receipt.json')
            assert trained['optimizer_updates']==(1 if name=='smoke' else 5000)
            assert trained['execution']['device']=='cuda:0'
            assert trained['execution']['model_dtype']=='float32'
            assert trained['execution']['parameters_gradients_adam_moments_cuda']
            assert not trained['execution']['amp'] and not trained['execution']['tf32']
            assert restored['mode']=='restore' and not restored['execution']['cuda_initialized']
            assert trained['closure']==restored['closure']
        record['state']='completed'
    except BaseException as error:
        record.update(state='failed',error_type=type(error).__name__)
        raise
    finally:
        record.update(finished_at_utc=utc(),elapsed_seconds=time.perf_counter()-started)
        write(status,record)


if __name__=='__main__':
    main()
