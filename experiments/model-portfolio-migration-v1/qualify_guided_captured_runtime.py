"""Exercise actual archive source identity and payload access without a model."""
import hashlib
import os
from pathlib import Path
import tempfile
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from guided_captured_runtime import GuidedCapturedRuntime


def run():
    path=DOC/'reports/guided-captured-runtime-qualification-01.json'
    require(not path.exists(), 'Preserve original source-view qualification')
    contract_path=RUN/'guided-component-contract-01.json';contract=read(contract_path)
    rows=[]
    for item in contract['models']:
        meta=read(ROOT/item['manifest']['path'])
        with tempfile.TemporaryDirectory(prefix='guided-source-access-fixture-',dir=RUN) as job:
            job=Path(job);output=job/'outputs';output.mkdir();fixture=job/'read-only-material.bin'
            fixture.write_bytes(b'literal task-owned material\n');original=fixture.read_bytes()
            with GuidedCapturedRuntime(item,meta,contract,output,extra_read_paths=[str(fixture.relative_to(ROOT))]) as runtime:
                actual_root=runtime.root
                source=runtime.root/'experiments/full-ai-data-diversity-v1/support.py'
                require(not source.is_symlink() and hashlib.sha256(source.read_bytes()).hexdigest()
                        ==meta['source_manifest']['members'][str(source.relative_to(runtime.root))], 'Actual regular captured Python identity required')
                same_inode=0
                for name,spec in item['non_Python_keepers'].items():
                    linked=runtime.root/name;keeper=ROOT/spec['path']
                    require(linked.is_symlink() and linked.resolve()==keeper.resolve()
                            and linked.stat().st_ino==keeper.stat().st_ino, 'Scientific payload must reuse exact original inode')
                    same_inode+=1
                require(fixture.read_bytes()==original, 'Explicit in-place material reads must work')
                rejects=[]
                for label,operation in [
                    ('unbound ambient read',lambda:(ROOT/'CONTEXT.md').read_bytes()),
                    ('traversal to unbound ambient file',lambda:(runtime.root/os.path.relpath(ROOT/'CONTEXT.md',runtime.root)).read_bytes()),
                    ('write to bound task-owned material',lambda:fixture.open('ab'))]:
                    try:
                        value=operation()
                    except PermissionError as error:
                        require(str(error).startswith('Unbound ambient scientific access in captured guided runtime:'), 'Exact intended source-view diagnostic required')
                        rejects.append(dict(case=label,type=type(error).__name__,diagnostic=str(error)))
                    else:
                        if hasattr(value,'close'):value.close()
                        raise AssertionError('Required rejection absent: '+label)
                (output/'literal.txt').write_text('authorized task output\n')
                require((output/'literal.txt').read_text()=='authorized task output\n', 'Owned output writes/reads must work')
                require(fixture.read_bytes()==original and len(rejects)==3, 'Actual negative controls must preserve fixture bytes')
                rows.append(dict(fit_item_id=item['fit_item_id'],captured_regular_python_exact=True,
                    scientific_payload_inodes_reused=same_inode,temporary_python_bytes=runtime.code_bytes,
                    explicit_material_read=True,owned_output_write=True,negative_controls=rejects))
            require(not actual_root.exists(), 'Owned temporary source view must be removed')
            rows[-1]['temporary_view_removed']=True
    save(path,dict(status='both_actual_guided_source_views_and_access_boundaries_qualified',created_unix=time.time(),
        producer=binding(Path(__file__)),runtime_helper=binding(Path(__file__).with_name('guided_captured_runtime.py')),
        contract=binding(contract_path),models=rows,negative_controls_passed=6,
        model_construction_or_inference=False,CPU_neural_forwards=0,CUDA_forwards=0,
        model_dataset_or_project_copies=0,temporary_Python_source_only=True,native_mutations=0,
        accuracy_qualified=False,latency_qualified=False,full_portfolio_complete=False))
    print('Both real source views pass six access controls;scientific inodes reused;temporary Python removed.',flush=True)


if __name__=='__main__':run()
