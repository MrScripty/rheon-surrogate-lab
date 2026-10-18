"""Focused verification: published content reuse rejects changed local bytes."""
import hashlib
from pathlib import Path
import tempfile
import importlib.util

spec = importlib.util.spec_from_file_location('tail_manage', Path(__file__).with_name('manage.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in ['first', 'second', 'third']:
            (root/name).mkdir()
        calls = []
        def download(reader, run, row, target):
            calls.append(run['id'])
            (target/row['name']).write_bytes(b'qualified bytes')
            return {'verification':'whole_sha256_verified'}
        receipts = []
        collect = module.cached_collector(download, receipts)
        row = dict(name='weights.npz', bytes=15, sha256=hashlib.sha256(b'qualified bytes').hexdigest())
        collect(None, {'id':'train'}, row, root/'first')
        reused = collect(None, {'id':'restore'}, row, root/'second')
        assert calls == ['train'] and reused['run_id'] == 'restore' and reused['source_run_id'] == 'train'
        (root/'second/weights.npz').write_bytes(b'corrupted bytes')
        try:
            collect(None, {'id':'next'}, row, root/'third')
        except ValueError:
            pass
        else:
            raise AssertionError('Changed local bytes must be rejected')
        assert not (root/'third/weights.npz').exists()
    print('Verified artifact reuse and corruption rejection')


if __name__ == '__main__':
    check()
