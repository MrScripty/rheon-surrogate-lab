"""Large fitting input validation and wrong-custody/race controls."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from managed_input_stream_v1 import streamed_identity,validate_inputs


class ManagedInputStreamTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.inputs={};self.expected={}
        for name in ('manifest','normalization','runtime_manifest','part_00'):
            path=self.root/name;self.inputs[name]=path
            count=70 if name=='part_00' else 1
            chunk=b'known captured input\x00'*32768;digest=hashlib.sha256()
            with path.open('wb') as handle:
                for _ in range(count):handle.write(chunk);digest.update(chunk)
            self.expected[name]=digest.hexdigest()
        self.evidence={name:dict(reference_id=name,measured_sha256=self.expected[name],
            project_id='a'*32,manifest_sha256='b'*64) for name in self.inputs}
        self.evidence_path=self.root/'evidence.json'
        self.args=SimpleNamespace(mode='train',input_evidence=self.evidence_path,
            manifest=self.inputs['manifest'],normalization=self.inputs['normalization'],
            runtime_manifest=self.inputs['runtime_manifest'],shards=[self.inputs['part_00']])

    def publish(self):self.evidence_path.write_text(json.dumps(self.evidence))

    def test_complete_typed_inputs_use_bounded_reads_and_exact_hashes(self):
        self.assertGreater(self.inputs['part_00'].stat().st_size,40*1024**2)
        self.publish();sizes=[];read=os.read
        def measured(fd,size):sizes.append(size);return read(fd,size)
        with patch('managed_input_stream_v1.os.read',side_effect=measured):
            _,observed=validate_inputs(self.args)
        self.assertEqual({n:r['sha256'] for n,r in observed.items()},self.expected)
        self.assertGreater(len(sizes),10);self.assertLessEqual(max(sizes),1024**2)

    def test_changed_hash_and_mixed_project_custody_are_rejected(self):
        self.evidence['part_00']['measured_sha256']='0'*64;self.publish()
        with self.assertRaises(ValueError):validate_inputs(self.args)
        self.evidence['part_00']['measured_sha256']=self.expected['part_00']
        self.evidence['part_00']['project_id']='c'*32;self.publish()
        with self.assertRaises(ValueError):validate_inputs(self.args)

    def test_link_and_concurrent_same_size_rewrite_are_rejected(self):
        link=self.root/'alias';link.symlink_to(self.inputs['part_00'])
        with self.assertRaises(OSError):streamed_identity(link)
        read=os.read;changed=False
        def mutate(fd,size):
            nonlocal changed
            chunk=read(fd,size)
            if not changed:
                changed=True
                before=self.inputs['part_00'].stat()
                with self.inputs['part_00'].open('r+b') as handle:handle.write(b'altered')
                os.utime(self.inputs['part_00'],ns=(before.st_atime_ns,before.st_mtime_ns))
            return chunk
        with patch('managed_input_stream_v1.os.read',side_effect=mutate),self.assertRaises(ValueError):
            streamed_identity(self.inputs['part_00'])


if __name__=='__main__':unittest.main()
